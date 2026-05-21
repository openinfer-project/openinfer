//! Text-only tokenizer and prompt contract for Kimi-K2.6.
//!
//! The first Kimi-K2.6 crate only accepts text chat messages. Image and video
//! inputs are represented here so frontend code has one explicit rejection path.

use crate::tensor::{HeaderError, HeaderResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextRole {
    System,
    User,
    Assistant,
    Tool,
}

impl TextRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextMessage {
    pub role: TextRole,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl TextMessage {
    #[must_use]
    pub fn new(role: TextRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
        }
    }

    #[must_use]
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        Self {
            role: TextRole::Assistant,
            content: content.into(),
            reasoning_content: Some(reasoning_content.into()),
        }
    }

    pub fn from_parts(role: TextRole, parts: Vec<PromptPart>) -> HeaderResult<Self> {
        reject_multimodal_parts(&parts)?;
        let content = parts
            .into_iter()
            .filter_map(|part| match part {
                PromptPart::Text(text) => Some(text),
                PromptPart::Image { .. } | PromptPart::Video { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("");
        Ok(Self::new(role, content))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingMode {
    Enabled,
    Instant,
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::Enabled
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub thinking: ThinkingMode,
    pub preserve_thinking: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            thinking: ThinkingMode::Enabled,
            preserve_thinking: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedPrompt {
    pub prompt: String,
    pub message_count: usize,
    pub thinking: ThinkingMode,
    pub preserve_thinking: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizedPrompt {
    pub rendered: RenderedPrompt,
    pub token_ids: Vec<u32>,
}

pub trait KimiK2TextTokenizer {
    fn encode_text(&self, prompt: &RenderedPrompt) -> HeaderResult<Vec<u32>>;
    fn decode_text(&self, token_ids: &[u32]) -> HeaderResult<String>;

    fn encode_messages(
        &self,
        messages: &[TextMessage],
        options: &ChatTemplateOptions,
    ) -> HeaderResult<TokenizedPrompt> {
        let rendered = render_text_chat_prompt(messages, options)?;
        let token_ids = self.encode_text(&rendered)?;
        Ok(TokenizedPrompt {
            rendered,
            token_ids,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptPart {
    Text(String),
    Image { mime_type: Option<String> },
    Video { mime_type: Option<String> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalKind {
    Image,
    Video,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedMultimodalInput {
    pub kind: MultimodalKind,
    pub reason: String,
}

pub fn render_text_chat_prompt(
    messages: &[TextMessage],
    options: &ChatTemplateOptions,
) -> HeaderResult<RenderedPrompt> {
    validate_text_messages(messages)?;

    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        if options.preserve_thinking {
            if let Some(reasoning) = &message.reasoning_content {
                prompt.push_str("<think>");
                prompt.push_str(reasoning);
                prompt.push_str("</think>\n");
            }
        }
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }

    if options.add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
        if options.thinking == ThinkingMode::Instant {
            prompt.push_str("<think></think>\n");
        }
    }

    Ok(RenderedPrompt {
        prompt,
        message_count: messages.len(),
        thinking: options.thinking,
        preserve_thinking: options.preserve_thinking,
    })
}

pub fn reject_multimodal_parts(parts: &[PromptPart]) -> HeaderResult<()> {
    let rejected = parts.iter().find_map(|part| match part {
        PromptPart::Text(_) => None,
        PromptPart::Image { .. } => Some(MultimodalKind::Image),
        PromptPart::Video { .. } => Some(MultimodalKind::Video),
    });
    match rejected {
        Some(kind) => Err(reject_multimodal(kind).into()),
        None => Ok(()),
    }
}

pub fn reject_multimodal(kind: MultimodalKind) -> RejectedMultimodalInput {
    let media = match kind {
        MultimodalKind::Image => "image",
        MultimodalKind::Video => "video",
    };
    RejectedMultimodalInput {
        kind,
        reason: format!(
            "Kimi-K2.6 header only supports text input; {media} inputs are rejected before tokenization"
        ),
    }
}

impl From<RejectedMultimodalInput> for HeaderError {
    fn from(value: RejectedMultimodalInput) -> Self {
        HeaderError::Unsupported {
            message: value.reason,
        }
    }
}

fn validate_text_messages(messages: &[TextMessage]) -> HeaderResult<()> {
    for (idx, message) in messages.iter().enumerate() {
        if message.content.is_empty() && message.reasoning_content.is_none() {
            return Err(HeaderError::Shape {
                message: format!("text message {idx} is empty"),
            });
        }
    }
    Ok(())
}
