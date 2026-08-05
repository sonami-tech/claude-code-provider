//! Shared model catalog types for providers.
//!
//! Each provider that tracks an upstream client (Claude Code, grok-shell, Codex
//! CLI) pins exactly one client version at compile time. The catalog *data*
//! lives in the provider crate; core only defines the shared model-entry shape.
//! Multi-version selection was removed (issue #12): operators use the shipped
//! pin, or an older Omni release for older wire.

/// One model in a provider catalog: a real upstream id plus inbound-only aliases.
///
/// Aliases are accepted on the way in but are not part of the advertised model
/// surface (mirrors how the providers already treat aliases).
#[derive(Debug, Clone, Copy)]
pub struct CatalogModel {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
}

impl CatalogModel {
    pub const fn new(id: &'static str, aliases: &'static [&'static str]) -> Self {
        Self { id, aliases }
    }

    /// True if `input` matches this model's id or any alias (case-sensitive, as
    /// upstream ids are).
    pub fn matches(&self, input: &str) -> bool {
        self.id == input || self.aliases.contains(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_is_case_sensitive_and_accepts_aliases() {
        // WHY: upstream model ids are case-sensitive; folding would send a wrong
        // id. Aliases must still resolve to membership for inbound routing.
        let m = CatalogModel::new("Grok", &["g"]);
        assert!(!m.matches("grok"));
        assert!(m.matches("Grok"));
        assert!(m.matches("g"));
        assert!(!m.matches("unknown"));
    }
}
