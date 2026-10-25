export default function ({ data, parentElement, setTriggerValue }) {
  // v2はデータ更新時にも関数を呼ぶ。前回のイベントとポータルを先に片付ける。
  parentElement.__shiruMenuCleanup?.();
  const commands = data?.commands || [];
  const effortOpen = Boolean(data?.effort_open);
  const efforts = [
    {command: "Low", mode: "low", description: "思考量を抑える"},
    {command: "Medium", mode: "medium", description: "通常の思考量"},
    {command: "High", mode: "high", description: "思考量を増やす"},
  ];
  const menu = document.createElement("div");
  const id = `slash-menu-${crypto.randomUUID()}`;
  menu.id = id;
  menu.className = "shiru-command-menu";
  menu.setAttribute("role", "listbox");
  menu.setAttribute("aria-label", effortOpen ? "effortの選択" : "コマンド候補");
  menu.hidden = true;
  const style = document.createElement("style");
  style.textContent = `
    .shiru-command-menu {
      position: fixed; z-index: 10000; padding: 6px;
      border: 1px solid #8886; border-radius: 12px;
      box-shadow: 0 6px 24px #0003; overflow-y: auto;
      box-sizing: border-box; font: inherit;
    }
    .shiru-command-menu[hidden] { display: none; }
    .shiru-command-menu button {
      display: flex; flex-wrap: wrap; gap: 4px 16px;
      width: 100%; padding: 10px 12px; border: 0; border-radius: 7px;
      text-align: left; background: transparent; color: inherit;
      font: inherit; cursor: pointer;
    }
    .shiru-command-menu button[aria-selected="true"] {
      background: color-mix(in srgb, currentColor 10%, transparent);
    }
    .shiru-command-menu strong { min-width: 140px; font-size: .9rem; }
    .shiru-command-menu span { font-size: .85rem; opacity: .75; }
    .shiru-command-menu footer { padding: 6px 12px 2px; font-size: .75rem; opacity: .65; }
  `;
  document.head.appendChild(style);
  document.body.appendChild(menu);
  let input = null;
  let matches = [];
  let selected = 0;
  let dismissedValue = null;
  let previousAttributes = {};

  let questionPanel = null;
  let questionReady = false;
  const attributes = ["aria-autocomplete", "aria-expanded", "aria-controls", "aria-activedescendant"];

  function hide() {
    menu.hidden = true;
    if (input) {
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
    }
  }

  function position() {
    const stop = document.querySelector('.st-key-execution_stop');
    const arrow = document.querySelector('.st-key-prompt_input [data-testid="stChatInputSubmitButton"]');
    if (stop && arrow && data?.running) {
      const rect = arrow.getBoundingClientRect();
      Object.assign(stop.style, {display: "block", left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px`});
    }
    if (!input || menu.hidden) return;
    const rect = input.closest('[data-testid="stChatInput"]').getBoundingClientRect();
    const theme = getComputedStyle(parentElement);
    const app = document.querySelector('[data-testid="stApp"]') || document.body;
    menu.style.backgroundColor = theme.getPropertyValue("--st-background-color").trim() || getComputedStyle(app).backgroundColor;
    menu.style.color = getComputedStyle(input).color;
    menu.style.left = `${rect.left}px`;
    menu.style.width = `${rect.width}px`;
    menu.style.bottom = `${window.innerHeight - rect.top + 8}px`;
    menu.style.maxHeight = `${Math.max(80, Math.min(390, rect.top - 16))}px`;
  }

  function highlight(index) {
    selected = index;
    const options = menu.querySelectorAll('[role="option"]');
    options.forEach((option, i) => option.setAttribute("aria-selected", String(i === selected)));
    if (options[selected]) {
      input.setAttribute("aria-activedescendant", options[selected].id);
      options[selected].scrollIntoView({ block: "nearest" });
    }
  }

  function complete(index, execute = true) {
    const choice = matches[index];
    if (!choice || !input) return;
    if (effortOpen) {
      hide();
      setTriggerValue("action", {kind: "effort", mode: choice.mode});
      return;
    }
    if (execute) {
      // サーバーのローカルコマンド処理へ渡し、モデルには送信しない。
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(input, "");
      input.dispatchEvent(new Event("input", {bubbles: true}));
      hide();
      setTriggerValue("action", {kind: "command", command: choice.command});
      return;
    }
    // Tabは入力補完のみ。
    dismissedValue = choice.command;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(input, choice.command);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();
    input.setSelectionRange(choice.command.length, choice.command.length);
    hide();
  }

  function refresh() {
    if (!input || (!effortOpen && document.activeElement !== input) || input.disabled) return hide();
    const value = input.value;
    const query = value.trimStart().toLowerCase();
    if (!effortOpen && (!query.startsWith("/") || query.includes("\n") || value === dismissedValue)) return hide();
    dismissedValue = null;
    matches = effortOpen ? efforts : commands.filter(item => item.command.startsWith(query));
    if (!matches.length) return hide();
    menu.replaceChildren();
    matches.forEach((item, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.tabIndex = -1;
      button.id = `${id}-${i}`;
      button.setAttribute("role", "option");
      const name = document.createElement("strong");
      name.textContent = (effortOpen ? `${i + 1}. ` : "") + item.command + (effortOpen && item.mode === data.effort ? " ✓" : "");
      const description = document.createElement("span");
      description.textContent = item.description;
      button.append(name, description);
      button.addEventListener("pointerdown", event => event.preventDefault());
      button.addEventListener("click", () => complete(i));
      button.addEventListener("pointermove", () => { if (selected !== i) highlight(i); });
      menu.appendChild(button);
    });
    const hint = document.createElement("footer");
    hint.textContent = effortOpen ? "1〜3で即確定 · ↑↓とEnterでも選択 · Escで閉じる" : "↑↓で選択 · Enterで確定 · Escで閉じる";
    menu.appendChild(hint);
    menu.hidden = false;
    input.setAttribute("aria-expanded", "true");
    position();
    highlight(effortOpen ? Math.max(0, efforts.findIndex(item => item.mode === data.effort)) : 0);
  }

  function keydown(event) {
    const editing = event.target instanceof Element && event.target.closest('input, textarea, [contenteditable="true"]');
    if (effortOpen && !menu.hidden && /^[1-3]$/.test(event.key)
        && (!editing || event.target === input || menu.contains(event.target))
        && !event.isComposing && event.keyCode !== 229
        && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
      event.preventDefault();
      event.stopImmediatePropagation();
      complete(Number(event.key) - 1);
      return;
    }
    if (!editing && !effortOpen && questionPanel && !event.isComposing && event.keyCode !== 229
        && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
      const buttons = [...questionPanel.querySelectorAll('[class*="_option_"] button:not(:disabled)')];
      const panel = [...questionPanel.classList].find(name => name.startsWith("st-key-clarification_picker_"))?.slice(7);
      if (buttons.length && (["ArrowDown", "ArrowUp", "Enter"].includes(event.key) || /^[1-5]$/.test(event.key))) {
        const current = buttons.indexOf(document.activeElement);
        const digit = /^[1-5]$/.test(event.key) ? Number(event.key) - 1 : null;
        // 他のボタンにフォーカスがある場合のEnterは、そのボタンの操作を優先する。
        if (event.key === "Enter" && current < 0 && document.activeElement?.closest("button")) return;
        if (digit !== null && digit >= buttons.length) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (event.key.startsWith("Arrow")) {
          const next = current < 0 ? 0 : (current + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
          buttons[next].focus();
        } else if (digit !== null) {
          setTriggerValue("action", {kind: "clarification", panel, index: digit});
        } else if (panel.includes("_single_") && current >= 0) {
          setTriggerValue("action", {kind: "clarification", panel, index: current});
        } else {
          setTriggerValue("action", {kind: "clarification", panel, submit: true});
        }
        return;
      }
    }
    // 空のチャット入力や画面上のEnterは、説明パネルを閉じる操作にする。
    // 入力済みの質問、自由記述の改行、候補の確定は通常の処理へ渡す。
    if (data?.command_output_open && menu.hidden && event.key === "Enter"
        && !event.isComposing && event.keyCode !== 229
        && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey
        && ((event.target === input && !input.value.trim())
            || (!editing && !event.target.closest?.("button")))) {
      event.preventDefault();
      event.stopImmediatePropagation();
      setTriggerValue("action", {kind: "close_command_output"});
      return;
    }
    if ((event.target !== input && !menu.contains(event.target)) || menu.hidden || event.isComposing || event.keyCode === 229) return;
    if (!["ArrowDown", "ArrowUp", "Enter", "Tab", "Escape"].includes(event.key)) return;
    if (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.key === "Escape") {
      dismissedValue = input.value;
      hide();
      if (effortOpen) setTriggerValue("action", {kind: "close_effort"});
    } else if (event.key === "ArrowDown") {
      highlight((selected + 1) % matches.length);
    } else if (event.key === "ArrowUp") {
      highlight((selected - 1 + matches.length) % matches.length);
    } else {
      complete(selected, event.key !== "Tab");
    }
  }

  function blur() {
    hide();
    if (effortOpen) setTriggerValue("action", {kind: "close_effort"});
  }
  function detach() {
    if (!input) return;
    input.removeEventListener("input", refresh);
    input.removeEventListener("focus", refresh);
    input.removeEventListener("blur", blur);
    for (const attr of attributes) {
      if (previousAttributes[attr] === null) input.removeAttribute(attr);
      else input.setAttribute(attr, previousAttributes[attr]);
    }
    input = null;
  }
  function attach() {
    position();
    const question = document.querySelector('[class*="st-key-clarification_picker_"]');
    if (question !== questionPanel) { questionPanel = question; questionReady = false; }
    if (question && !questionReady && !effortOpen) {
      const button = question.querySelector('[class*="_option_"] button:not(:disabled)');
      if (button) { button.focus(); questionReady = true; }
    }
    const next = document.querySelector(".st-key-prompt_input textarea");
    if (next === input) return;
    hide();
    detach();
    input = next;
    dismissedValue = null;
    if (!input) return;
    previousAttributes = Object.fromEntries(attributes.map(attr => [attr, input.getAttribute(attr)]));
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-controls", id);
    input.addEventListener("input", refresh);
    input.addEventListener("focus", refresh);
    input.addEventListener("blur", blur);
    if (effortOpen) input.focus();
    refresh();
  }
  const observer = new MutationObserver(attach);
  observer.observe(document.body, { childList: true, subtree: true });
  document.addEventListener("keydown", keydown, true);
  window.addEventListener("resize", position);
  window.addEventListener("scroll", position, true);
  window.visualViewport?.addEventListener("resize", position);
  attach();
  const cleanup = () => {
    observer.disconnect();
    document.removeEventListener("keydown", keydown, true);
    window.removeEventListener("resize", position);
    window.removeEventListener("scroll", position, true);
    window.visualViewport?.removeEventListener("resize", position);
    detach();
    menu.remove();
    style.remove();
  };
  parentElement.__shiruMenuCleanup = cleanup;
  return () => {
    cleanup();
    if (parentElement.__shiruMenuCleanup === cleanup) delete parentElement.__shiruMenuCleanup;
  };
}
