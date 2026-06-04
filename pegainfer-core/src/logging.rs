//! Unified logging configuration.

use std::sync::Once;

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;

static INIT: Once = Once::new();

#[derive(Debug, Clone)]
struct LoggingConfig {
    level: String,
    colored: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            colored: true,
        }
    }
}

const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 5] = [
    ("h2", "warn"),
    ("hyper", "warn"),
    ("hyper_util", "warn"),
    ("axum", "warn"),
    ("tower", "warn"),
];

fn apply_default_module_levels(mut filter: String) -> String {
    for (module, level) in DEFAULT_NOISY_MODULE_LEVELS {
        let module_pattern = format!("{module}=");
        if !filter.contains(&module_pattern) {
            if !filter.is_empty() {
                filter.push(',');
            }
            filter.push_str(module);
            filter.push('=');
            filter.push_str(level);
        }
    }
    filter
}

fn resolved_filter_string(level: String, rust_log: Option<String>) -> String {
    apply_default_module_levels(rust_log.unwrap_or(level))
}

fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig { level, colored } = config;
        let filter_str = resolved_filter_string(level, std::env::var("RUST_LOG").ok());
        let filter =
            logforth::filter::env_filter::EnvFilterBuilder::from_env_or("RUST_LOG", filter_str)
                .build();

        let mut builder = logforth::starter_log::builder();
        if colored {
            let layout = TextLayout::default()
                .info_color(Green)
                .warn_color(Yellow)
                .error_color(Red);
            builder = builder.dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default().with_layout(layout))
            });
        } else {
            builder = builder.dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default())
            });
        }

        builder.apply();
    });
}

pub fn init_default() {
    init(LoggingConfig::default());
}

#[cfg(test)]
mod tests {
    use super::resolved_filter_string;

    #[test]
    fn applies_default_noisy_module_levels_when_rust_log_is_debug() {
        let filter = resolved_filter_string("info".to_string(), Some("debug".to_string()));

        assert!(filter.starts_with("debug"));
        assert!(filter.contains("h2=warn"));
        assert!(filter.contains("hyper=warn"));
        assert!(filter.contains("hyper_util=warn"));
        assert!(filter.contains("axum=warn"));
        assert!(filter.contains("tower=warn"));
    }

    #[test]
    fn preserves_explicit_module_overrides_from_rust_log() {
        let filter = resolved_filter_string("info".to_string(), Some("debug,h2=trace".to_string()));

        assert!(filter.starts_with("debug,h2=trace"));
        assert!(filter.contains("hyper=warn"));
        assert!(filter.contains("hyper_util=warn"));
        assert!(filter.contains("axum=warn"));
        assert!(filter.contains("tower=warn"));
        assert!(!filter.contains("h2=warn"));
    }
}
