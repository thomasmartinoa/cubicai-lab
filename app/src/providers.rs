//! Where chats go. Everything speaks the OpenAI chat format; "This PC" is the llama-server the
//! Engine Room runs.

pub struct Provider {
    pub id: &'static str,
    pub name: &'static str,
    pub short: &'static str,
    pub base_url: &'static str,
    pub needs_key: bool,
    pub key_url: &'static str,
    pub suggested: &'static [&'static str],
}

pub const PC: &str = "pc";

pub const ALL: &[Provider] = &[
    Provider {
        id: PC,
        name: "This PC (llama.cpp)",
        short: "This PC",
        base_url: "",
        needs_key: false,
        key_url: "",
        suggested: &[],
    },
    Provider {
        id: "ollama",
        name: "Ollama on this PC",
        short: "Ollama",
        base_url: "http://127.0.0.1:11434/v1",
        needs_key: false,
        key_url: "",
        suggested: &[],
    },
    Provider {
        id: "gemini",
        name: "Google Gemini",
        short: "Gemini",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        needs_key: true,
        key_url: "https://aistudio.google.com/apikey",
        suggested: &["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview"],
    },
    Provider {
        id: "xai",
        name: "xAI (Grok)",
        short: "Grok",
        base_url: "https://api.x.ai/v1",
        needs_key: true,
        key_url: "https://console.x.ai",
        suggested: &["grok-4.7", "grok-4.6", "grok-4.3"],
    },
    Provider {
        id: "openrouter",
        name: "OpenRouter",
        short: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        needs_key: true,
        key_url: "https://openrouter.ai/keys",
        suggested: &["google/gemma-4-31b-it:free", "openrouter/free", "x-ai/grok-4.7"],
    },
    Provider {
        id: "custom",
        name: "Custom (OpenAI-compatible)",
        short: "Custom",
        base_url: "",
        needs_key: true,
        key_url: "",
        suggested: &[],
    },
];

pub fn get(id: &str) -> &'static Provider {
    ALL.iter().find(|p| p.id == id).unwrap_or(&ALL[0])
}
