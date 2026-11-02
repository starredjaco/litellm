mod credentials;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_config::{ConfiguredRedis, Resolution, SecretSnapshot, Value};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::PythonContext;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{CredentialSource, HostServices, SharedEnvironment};
use litellm_redis_types::ResolvedRedis;
use pyo3::{
    prelude::*,
    sync::MutexExt,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use self::credentials::PythonCredentialSource;
use crate::redis::events::{ServiceLoggingEvents, state_label};

/// The `redis_kwargs` dict `RedisCache` hands redis-py, read the way `litellm/_redis.py` reads it.
/// A `None` stays an explicit null that hides its `REDIS_*` value, as in redis-py. A value that is not
/// plain data reaches the resolver by its type name, so the resolver or the connector names the
/// setting it cannot honor. A `credential_provider` object is also kept, so the connector can ask it
/// for credentials.
pub(crate) struct PythonRedisSettings {
    settings: ConfiguredRedis,
    credential_provider: Option<Py<PyAny>>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for PythonRedisSettings {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let kwargs = kwargs.cast::<PyDict>()?;
        let plain = PyDict::new(py);
        for (name, value) in kwargs.iter() {
            if value.is_none() || is_plain(&value) {
                plain.set_item(&name, value)?;
            } else {
                plain.set_item(&name, value.get_type().name()?)?;
            }
        }
        let credential_provider = kwargs
            .get_item("credential_provider")?
            .filter(|provider| !provider.is_none() && !is_plain(provider))
            .map(Bound::unbind);
        let fields: BTreeMap<String, Value> = litellm_host_python::from_py_argument(plain.as_any())?;
        Ok(Self {
            settings: ConfiguredRedis::redis_cache_kwargs(fields),
            credential_provider,
        })
    }
}

fn is_plain(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyTuple>()
        || value.is_instance_of::<PyDict>()
}

struct Shared {
    resolved: ResolvedRedis,
    credential_provider: Option<Py<PyAny>>,
    client: RedisClient,
}

impl Shared {
    fn serves(&self, resolved: &ResolvedRedis, credential_provider: Option<&Py<PyAny>>) -> bool {
        let same_provider = match (&self.credential_provider, credential_provider) {
            (None, None) => true,
            (Some(shared), Some(provider)) => shared.is(provider),
            _ => false,
        };
        same_provider && self.resolved == *resolved
    }
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The one client for a resolved Redis config and credential provider, so every family in this process shares its pool
/// and its breaker. Like redis-py, which never connects at construction, a config the resolver or
/// the connector cannot honor leaves the family on memory with a warning rather than failing the
/// `Router`.
fn shared_client(
    py: Python<'_>,
    kwargs: PythonRedisSettings,
    namespace: Option<String>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let PythonRedisSettings {
        settings,
        credential_provider,
    } = kwargs;
    let secrets = match secret_snapshot(py, settings.secret_names())? {
        Ok(secrets) => secrets,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| settings.resolve(&secrets, &*environment)) {
        Ok(Resolution { redis, ignored }) => {
            logger.scope(|| {
                for ignored in ignored {
                    litellm_tracing::warn!(
                        "Redis setting {} is not supported by the Rust client and is ignored: {}",
                        ignored.setting,
                        ignored.consequence
                    );
                }
            });
            ResolvedRedis { namespace, ..redis }
        }
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let breaker = match logger.scope(|| litellm_redis_connect::breaker_config(&*environment)) {
        Ok(breaker) => breaker,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let runtime = litellm_host_python::runtime_handle()?;
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py, runtime.clone())?);
    let host = HostServices {
        events: Some(events),
        breaker,
        http: crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        environment,
        credential_provider: credential_provider
            .as_ref()
            .map(|provider| -> PyResult<Arc<dyn CredentialSource>> {
                Ok(Arc::new(PythonCredentialSource::new(
                    provider.clone_ref(py),
                    PythonContext::capture(py)?,
                )))
            })
            .transpose()?,
    };
    let mut clients = CLIENTS
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(shared) = clients
        .iter()
        .find(|shared| shared.serves(&resolved, credential_provider.as_ref()))
    {
        return Ok(Some(shared.client.clone()));
    }
    let connected = {
        let _runtime = runtime.enter();
        litellm_redis_connect::connect(resolved.clone(), host)
    };
    let client = match connected {
        Ok(client) => client,
        Err(error) => {
            drop(clients);
            return Ok(memory_only(py, &error));
        }
    };
    clients.push(Shared {
        resolved,
        credential_provider,
        client: client.clone(),
    });
    Ok(Some(client))
}

/// The Rust client for one `RedisCache`, built once and handed to every native store, so a store
/// never sees the redis-py kwargs or the namespace.
#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient(pub(crate) RedisClient);

#[pymethods]
impl NativeRedisClient {
    /// `None` when Redis is not usable, which leaves every store given it on memory.
    #[staticmethod]
    #[pyo3(signature = (redis_kwargs, *, namespace = None))]
    fn connect(
        py: Python<'_>,
        redis_kwargs: PythonRedisSettings,
        namespace: Option<String>,
    ) -> PyResult<Option<Self>> {
        Ok(shared_client(py, redis_kwargs, namespace)?.map(Self))
    }
}

/// The breaker state of every native Redis client this process built, so the Prometheus state
/// gauge counts them next to the Python `RedisCache` breakers.
#[pyfunction]
pub(crate) fn native_redis_breaker_states(py: Python<'_>) -> Vec<&'static str> {
    let clients = CLIENTS
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    clients
        .iter()
        .filter_map(|shared| shared.client.breaker_state())
        .map(|state| state_label(&state))
        .collect()
}

/// Every name the resolver may read, fetched once the way Python's `get_secret` fetches it: the
/// configured secret manager first, then the process environment.
fn secret_snapshot(
    py: Python<'_>,
    names: BTreeSet<String>,
) -> PyResult<Result<SecretSnapshot, litellm_secrets::Error>> {
    let source = crate::secrets::source(py)?;
    let owned: Vec<String> = names.iter().cloned().collect();
    let lookup = crate::execution::run_sync_value(py, async move {
        let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
        Ok(source.resolve(&borrowed).await)
    })?;
    Ok(lookup.map(|lookup| SecretSnapshot::read(names, &*lookup)))
}

fn memory_only(py: Python<'_>, error: &(dyn std::error::Error + 'static)) -> Option<RedisClient> {
    let report = litellm_error::Report::new(error).to_string();
    crate::logger::capture(py).scope(|| {
        litellm_tracing::warn!(
            "Redis is not usable from the Rust stores, their state stays in memory: {report}"
        )
    });
    None
}
