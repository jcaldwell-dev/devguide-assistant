"use strict";

const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");

const chatMessages = document.getElementById("chatMessages");

const newConversationButton = document.getElementById(
    "newConversationButton"
);

const clearConversationButton = document.getElementById(
    "clearConversationButton"
);

const conversationIdDisplay = document.getElementById(
    "conversationIdDisplay"
);

const connectionStatus = document.getElementById(
    "connectionStatus"
);

const modelDisplay = document.getElementById("modelDisplay");

const activityPanel = document.getElementById("activityPanel");
const toolActivity = document.getElementById("toolActivity");

const metricsPanel = document.getElementById("metricsPanel");

const modelRoundsMetric = document.getElementById(
    "modelRoundsMetric"
);

const inputTokensMetric = document.getElementById(
    "inputTokensMetric"
);

const outputTokensMetric = document.getElementById(
    "outputTokensMetric"
);

const totalTokensMetric = document.getElementById(
    "totalTokensMetric"
);


let conversationId = createConversationId();
let requestInProgress = false;


function createConversationId() {
    if (
        window.crypto
        && typeof window.crypto.randomUUID === "function"
    ) {
        return window.crypto.randomUUID();
    }

    return (
        `conversation-${Date.now()}-`
        + Math.random().toString(16).slice(2)
    );
}


function updateConversationDisplay() {
    conversationIdDisplay.textContent = conversationId;
}


function setStatus(text, statusClass) {
    connectionStatus.textContent = text;

    connectionStatus.classList.remove(
        "status-ready",
        "status-working",
        "status-error"
    );

    connectionStatus.classList.add(statusClass);
}


function setRequestState(isActive) {
    requestInProgress = isActive;

    sendButton.disabled = isActive;
    messageInput.disabled = isActive;

    newConversationButton.disabled = isActive;
    clearConversationButton.disabled = isActive;

    sendButton.textContent = isActive
        ? "Receiving..."
        : "Send";
}


function scrollChatToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function createMessage(role, initialText = "") {
    const article = document.createElement("article");

    article.classList.add("message");

    if (role === "user") {
        article.classList.add("user-message");
    } else if (role === "error") {
        article.classList.add("error-message");
    } else {
        article.classList.add("assistant-message");
    }

    const label = document.createElement("div");
    label.classList.add("message-label");

    label.textContent = role === "user"
        ? "You"
        : role === "error"
            ? "Error"
            : "Assistant";

    const content = document.createElement("div");
    content.classList.add("message-content");
    content.textContent = initialText;

    article.append(label, content);
    chatMessages.appendChild(article);

    scrollChatToBottom();

    return {
        article,
        content,
    };
}


function resetTurnPanels() {
    toolActivity.replaceChildren();

    activityPanel.classList.add("hidden");
    metricsPanel.classList.add("hidden");

    modelRoundsMetric.textContent = "0";
    inputTokensMetric.textContent = "0";
    outputTokensMetric.textContent = "0";
    totalTokensMetric.textContent = "0";
}


function addToolActivity(toolCall) {
    activityPanel.classList.remove("hidden");

    const entry = document.createElement("div");
    entry.classList.add("tool-entry");

    const title = document.createElement("strong");

    const status = toolCall.success
        ? "completed"
        : `failed: ${toolCall.error_type || "unknown"}`;

    title.textContent = `${toolCall.tool_name} (${status})`;

    const argumentsCode = document.createElement("code");

    argumentsCode.textContent = JSON.stringify(
        toolCall.arguments,
        null,
        2
    );

    entry.append(title, argumentsCode);
    toolActivity.appendChild(entry);
}


function updateMetrics(eventData) {
    metricsPanel.classList.remove("hidden");

    modelDisplay.textContent = eventData.model || "Unknown";

    modelRoundsMetric.textContent = String(
        eventData.model_rounds ?? 0
    );

    inputTokensMetric.textContent = String(
        eventData.input_tokens ?? 0
    );

    outputTokensMetric.textContent = String(
        eventData.output_tokens ?? 0
    );

    totalTokensMetric.textContent = String(
        eventData.total_tokens ?? 0
    );
}


function parseSseBlock(block) {
    const lines = block.split(/\r?\n/);

    let eventName = "message";
    const dataLines = [];

    for (const line of lines) {
        if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
        }

        if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    }

    if (dataLines.length === 0) {
        return null;
    }

    return {
        eventName,
        data: JSON.parse(dataLines.join("\n")),
    };
}


async function consumeSseStream(response, onEvent) {
    if (!response.body) {
        throw new Error(
            "The browser did not provide a readable response body."
        );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();

        if (done) {
            buffer += decoder.decode();
        } else {
            buffer += decoder.decode(
                value,
                { stream: true }
            );
        }

        const blocks = buffer.split(/\r?\n\r?\n/);

        buffer = blocks.pop() || "";

        for (const block of blocks) {
            if (!block.trim()) {
                continue;
            }

            const parsedEvent = parseSseBlock(block);

            if (parsedEvent) {
                onEvent(
                    parsedEvent.eventName,
                    parsedEvent.data
                );
            }
        }

        if (done) {
            break;
        }
    }

    if (buffer.trim()) {
        const parsedEvent = parseSseBlock(buffer);

        if (parsedEvent) {
            onEvent(
                parsedEvent.eventName,
                parsedEvent.data
            );
        }
    }
}


async function sendMessage(userMessage) {
    if (requestInProgress) {
        return;
    }

    resetTurnPanels();

    createMessage("user", userMessage);

    const assistantMessage = createMessage("assistant");
    assistantMessage.content.classList.add("streaming-cursor");

    setRequestState(true);
    setStatus("Streaming", "status-working");

    let fullAssistantText = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                conversation_id: conversationId,
                user_message: userMessage,
            }),
        });

        if (!response.ok) {
            let errorMessage = (
                `Request failed with status ${response.status}.`
            );

            try {
                const errorBody = await response.json();

                if (errorBody.detail) {
                    errorMessage = errorBody.detail;
                }
            } catch {
                // Leave the fallback error message unchanged.
            }

            throw new Error(errorMessage);
        }

        await consumeSseStream(
            response,
            (eventName, eventData) => {
                if (eventName === "start") {
                    modelDisplay.textContent = (
                        eventData.model || "Unknown"
                    );
                }

                if (eventName === "tool_call") {
                    addToolActivity(eventData.tool_call);
                }

                if (eventName === "text_delta") {
                    const delta = eventData.delta || "";

                    fullAssistantText += delta;

                    assistantMessage.content.textContent = (
                        fullAssistantText
                    );

                    scrollChatToBottom();
                }

                if (eventName === "complete") {
                    if (
                        !fullAssistantText
                        && eventData.text
                    ) {
                        fullAssistantText = eventData.text;

                        assistantMessage.content.textContent = (
                            fullAssistantText
                        );
                    }

                    updateMetrics(eventData);
                    setStatus("Ready", "status-ready");
                }

                if (eventName === "error") {
                    throw new Error(
                        eventData.error_message
                        || "The streamed request failed."
                    );
                }
            }
        );

        if (!fullAssistantText) {
            throw new Error(
                "The assistant returned no visible text."
            );
        }

    } catch (error) {
        assistantMessage.article.remove();

        createMessage(
            "error",
            error instanceof Error
                ? error.message
                : String(error)
        );

        setStatus("Error", "status-error");

    } finally {
        assistantMessage.content.classList.remove(
            "streaming-cursor"
        );

        setRequestState(false);
        messageInput.focus();
    }
}


async function clearCurrentConversation() {
    if (requestInProgress) {
        return;
    }

    setStatus("Clearing", "status-working");

    try {
        const response = await fetch(
            `/conversations/${encodeURIComponent(
                conversationId
            )}`,
            {
                method: "DELETE",
            }
        );

        if (!response.ok) {
            throw new Error(
                `Clear request failed with status `
                + `${response.status}.`
            );
        }

        chatMessages.replaceChildren();

        createMessage(
            "assistant",
            "Conversation memory has been cleared."
        );

        resetTurnPanels();
        modelDisplay.textContent = "Not used yet";

        setStatus("Ready", "status-ready");

    } catch (error) {
        createMessage(
            "error",
            error instanceof Error
                ? error.message
                : String(error)
        );

        setStatus("Error", "status-error");
    }
}


function startNewConversation() {
    if (requestInProgress) {
        return;
    }

    conversationId = createConversationId();
    updateConversationDisplay();

    chatMessages.replaceChildren();

    createMessage(
        "assistant",
        "A new conversation has started."
    );

    resetTurnPanels();
    modelDisplay.textContent = "Not used yet";

    setStatus("Ready", "status-ready");

    messageInput.focus();
}


chatForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const userMessage = messageInput.value.trim();

        if (!userMessage) {
            return;
        }

        messageInput.value = "";

        await sendMessage(userMessage);
    }
);


messageInput.addEventListener(
    "keydown",
    (event) => {
        if (
            event.key === "Enter"
            && !event.shiftKey
        ) {
            event.preventDefault();
            chatForm.requestSubmit();
        }
    }
);


newConversationButton.addEventListener(
    "click",
    startNewConversation
);


clearConversationButton.addEventListener(
    "click",
    clearCurrentConversation
);


updateConversationDisplay();
messageInput.focus();