//! Zero-hot-path-allocation JSON localization (`lang/en.json`, `lang/ko.json`).
//!
//! # Real-Time Guarantees
//! - JSON files are parsed ONLY on startup or when the user changes language in Settings.
//! - External `lang/{code}.json` overrides are checked once during `I18n::load`, falling back
//!   to embedded `include_str!` defaults so the binary works even without external `lang/` files.
//! - Per-frame UI lookups (`i18n.t("key")`) perform a single hash lookup returning `&str`
//!   with zero heap allocations and zero file I/O.

use std::path::PathBuf;

use rustc_hash::FxHashMap;

use crate::settings::{Language, MidiParsing, Statistics};

const EMBEDDED_EN_JSON: &str = include_str!("../lang/en.json");
const EMBEDDED_KO_JSON: &str = include_str!("../lang/ko.json");

pub struct I18n {
    language: Language,
    table: FxHashMap<String, String>,
    fallback: FxHashMap<String, String>,
}

impl I18n {
    pub fn load(language: Language) -> Self {
        let fallback = parse_lang_map(EMBEDDED_EN_JSON);
        let embedded = match language {
            Language::English => EMBEDDED_EN_JSON,
            Language::Korean => EMBEDDED_KO_JSON,
        };
        let mut table = parse_lang_map(embedded);

        // Allow optional external `lang/{code}.json` override next to executable or CWD
        if let Some(ext_json) = try_read_external_lang_file(language.code()) {
            let ext_map = parse_lang_map(&ext_json);
            for (k, v) in ext_map {
                table.insert(k, v);
            }
        }

        Self {
            language,
            table,
            fallback,
        }
    }

    pub fn set_language(&mut self, language: Language) {
        if self.language != language {
            *self = Self::load(language);
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn language(&self) -> Language {
        self.language
    }

    /// Zero-allocation lookup returning the localized string (or English fallback, or key itself).
    #[inline]
    pub fn t<'a>(&'a self, key: &'a str) -> &'a str {
        if let Some(val) = self.table.get(key) {
            return val.as_str();
        }
        if let Some(val) = self.fallback.get(key) {
            return val.as_str();
        }
        key
    }

    /// Localized label for a [`Statistics`] row in Visual settings.
    #[inline]
    pub fn stat_name(&self, stat: Statistics) -> &str {
        match stat {
            Statistics::Time => self.t("hud.stat.time"),
            Statistics::Fps => self.t("hud.stat.fps"),
            Statistics::VoiceCount => self.t("hud.stat.voice"),
            Statistics::Rendered => self.t("hud.stat.rendered"),
            Statistics::NoteCount => self.t("hud.stat.note_count"),
            Statistics::Polyphony => self.t("hud.stat.polyphony"),
            Statistics::Nps => self.t("hud.stat.nps"),
            Statistics::Xrun => self.t("hud.stat.xrun"),
            Statistics::Cpu => self.t("hud.stat.cpu"),
            Statistics::Ram => self.t("hud.stat.ram"),
            Statistics::Gpu => self.t("hud.stat.gpu"),
        }
    }

    /// Localized label for a [`MidiParsing`] mode in MIDI settings.
    #[inline]
    pub fn parsing_name(&self, parsing: MidiParsing) -> &str {
        match parsing {
            MidiParsing::Cake => self.t("midi.parsing.cake"),
            MidiParsing::Ram => self.t("midi.parsing.ram"),
            MidiParsing::Live => self.t("midi.parsing.live"),
        }
    }
}

fn parse_lang_map(json_str: &str) -> FxHashMap<String, String> {
    serde_json::from_str::<FxHashMap<String, String>>(json_str).unwrap_or_default()
}

fn try_read_external_lang_file(code: &str) -> Option<String> {
    let filename = format!("{code}.json");
    let mut candidates: Vec<PathBuf> = Vec::with_capacity(2);
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("lang").join(&filename));
        }
    }
    candidates.push(PathBuf::from("lang").join(&filename));

    for path in candidates {
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                return Some(content);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn embedded_en_and_ko_json_have_exact_matching_keys() {
        let en: FxHashMap<String, String> =
            serde_json::from_str(EMBEDDED_EN_JSON).expect("parse lang/en.json");
        let ko: FxHashMap<String, String> =
            serde_json::from_str(EMBEDDED_KO_JSON).expect("parse lang/ko.json");

        assert!(!en.is_empty(), "en.json must not be empty");
        assert!(!ko.is_empty(), "ko.json must not be empty");

        let en_keys: BTreeSet<&String> = en.keys().collect();
        let ko_keys: BTreeSet<&String> = ko.keys().collect();

        let missing_in_ko: Vec<&&String> = en_keys.difference(&ko_keys).collect();
        let extra_in_ko: Vec<&&String> = ko_keys.difference(&en_keys).collect();
        assert!(
            missing_in_ko.is_empty(),
            "Keys missing in lang/ko.json: {missing_in_ko:?}"
        );
        assert!(
            extra_in_ko.is_empty(),
            "Extra keys in lang/ko.json not present in lang/en.json: {extra_in_ko:?}"
        );

        // Verify zero empty values
        for (k, v) in &en {
            assert!(!v.trim().is_empty(), "en.json key '{k}' is empty");
        }
        for (k, v) in &ko {
            assert!(!v.trim().is_empty(), "ko.json key '{k}' is empty");
        }

        let i18n_ko = I18n::load(Language::Korean);
        assert_eq!(i18n_ko.t("settings.title"), "설정");
        assert_eq!(i18n_ko.parsing_name(MidiParsing::Cake), "떡 (Rice Cake)");
        let i18n_en = I18n::load(Language::English);
        assert_eq!(i18n_en.t("settings.title"), "Settings");
        assert_eq!(i18n_en.parsing_name(MidiParsing::Cake), "Rice Cake");
    }
}
