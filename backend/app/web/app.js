const chatLog = document.querySelector("#chat-log");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newChatButton = document.querySelector("#new-chat");

let sessionId = createSessionId();

function createSessionId() {
  if (window.crypto?.randomUUID) return `web-${window.crypto.randomUUID()}`;
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function addTextMessage(role, text, extraClass = "") {
  const message = document.createElement("article");
  message.className = `message ${role}-message ${extraClass}`;
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "留学生咨询助手";
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  message.append(label, body);
  chatLog.append(message);
  scrollToLatest();
  return message;
}

function addAssistantResponse(payload) {
  const message = addTextMessage("assistant", payload.answer || "暂时无法生成回答，请稍后重试。");
  const meta = document.createElement("div");
  meta.className = "meta-grid";
  [
    payload.route && `路由：${payload.route}`,
    payload.metrics && `步骤：${payload.metrics.steps_used}`,
    payload.used_tools?.length && `工具：${payload.used_tools.join(", ")}`,
  ].filter(Boolean).forEach((label) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = label;
    meta.append(pill);
  });
  if (meta.childElementCount) message.append(meta);

  if (payload.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "source-list";
    const title = document.createElement("strong");
    title.textContent = "参考来源";
    const list = document.createElement("ul");
    payload.sources.forEach((source) => {
      const item = document.createElement("li");
      item.textContent = source;
      list.append(item);
    });
    sources.append(title, list);
    message.append(sources);
  }

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "查看 Agent 执行详情";
  const data = document.createElement("dl");
  data.className = "detail-list";
  const rows = [
    ["计划", payload.plan?.subgoals?.join(" → ") || "单步回答"],
    ["步骤路由", payload.steps?.map((step) => step.route).join(" → ") || payload.route || "无"],
    ["反思", payload.reflection ? `${payload.reflection.next_action} · ${payload.reflection.judge_source}` : "无"],
    ["评估", payload.evaluation ? `${payload.evaluation.score} · ${payload.evaluation.source}` : "无"],
  ];
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    row.append(term, description);
    data.append(row);
  });
  details.append(summary, data);
  message.append(details);

  if (payload.event_id) {
    addFeedbackControls(message, payload.event_id);
  }
}

function addFeedbackControls(message, eventId) {
  const feedback = document.createElement("div");
  feedback.className = "feedback";
  const prompt = document.createElement("span");
  prompt.textContent = "这条回答有帮助吗？";
  const helpful = document.createElement("button");
  helpful.type = "button";
  helpful.textContent = "有帮助";
  const notHelpful = document.createElement("button");
  notHelpful.type = "button";
  notHelpful.textContent = "没帮助";

  const submit = async (rating) => {
    helpful.disabled = true;
    notHelpful.disabled = true;
    try {
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_id: eventId, rating }),
      });
      if (!response.ok) throw new Error("反馈保存失败");
      prompt.textContent = "感谢你的反馈。";
      helpful.remove();
      notHelpful.remove();
    } catch (_) {
      prompt.textContent = "反馈暂未保存，请稍后重试。";
      helpful.disabled = false;
      notHelpful.disabled = false;
    }
  };
  helpful.addEventListener("click", () => submit("helpful"));
  notHelpful.addEventListener("click", () => submit("not_helpful"));
  feedback.append(prompt, helpful, notHelpful);
  message.append(feedback);
}

function scrollToLatest() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function setPending(pending) {
  sendButton.disabled = pending;
  input.disabled = pending;
  sendButton.textContent = pending ? "处理中…" : "发送";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addTextMessage("user", text);
  input.value = "";
  setPending(true);
  const loading = document.createElement("div");
  loading.className = "loading";
  loading.textContent = "正在规划并检索资料…";
  chatLog.append(loading);
  scrollToLatest();

  try {
    const response = await fetch("/agent-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-persist-experience": "true" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Agent 请求失败，请稍后重试。");
    addAssistantResponse(payload);
  } catch (error) {
    addTextMessage("assistant", error.message || "暂时无法连接助手，请稍后重试。", "error-message");
  } finally {
    loading.remove();
    setPending(false);
    input.focus();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.message || "";
    input.focus();
  });
});

newChatButton.addEventListener("click", () => {
  sessionId = createSessionId();
  chatLog.innerHTML = "";
  addTextMessage("assistant", "已开始新对话。你想咨询什么问题？");
  input.focus();
});
