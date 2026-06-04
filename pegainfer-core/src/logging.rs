//! Unified logging configuration.

use std::str::FromStr;
use std::sync::Once;

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use logforth::record::Level;

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

fn bare_global_level(filter: &str) -> Option<Option<Level>> {
    let mut global_level = None;

    for directive in filter
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if directive.contains('=') {
            continue;
        }

        if directive.eq_ignore_ascii_case("off") {
            global_level = Some(None);
        } else if let Ok(level) = Level::from_str(directive) {
            global_level = Some(Some(level));
        }
    }

    global_level
}

fn should_apply_default_module_levels(filter: &str) -> bool {
    matches!(bare_global_level(filter), Some(Some(level)) if level < Level::Warn)
}

fn resolved_filter_string(level: String, rust_log: Option<String>) -> String {
    let filter = rust_log.unwrap_or(level);
    if should_apply_default_module_levels(&filter) {
        apply_default_module_levels(filter)
    } else {
        filter
    }
}

fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig { level, colored } = config;
        let filter_str = resolved_filter_string(level, std::env::var("RUST_LOG").ok());
        let filter = logforth::filter::env_filter::EnvFilterBuilder::from_spec(filter_str).build();

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

    #[test]
    fn does_not_relax_stricter_global_rust_log_levels() {
        let error_filter = resolved_filter_string("info".to_string(), Some("error".to_string()));
        let off_filter = resolved_filter_string("info".to_string(), Some("off".to_string()));

        assert_eq!(error_filter, "error");
        assert_eq!(off_filter, "off");
    }

    #[test]
    fn does_not_add_default_module_levels_without_a_global_level() {
        let filter = resolved_filter_string("info".to_string(), Some("h2=trace".to_string()));

        assert_eq!(filter, "h2=trace");
        assert!(!filter.contains("hyper=warn"));
        assert!(!filter.contains("hyper_util=warn"));
        assert!(!filter.contains("axum=warn"));
        assert!(!filter.contains("tower=warn"));
    }
}
