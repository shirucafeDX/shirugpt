export default function ({ data, parentElement }) {
  const icon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg>';
  const button = document.createElement('button');
  button.type = 'button';
  button.title = '回答をコピー';
  button.setAttribute('aria-label', '回答をコピー');
  button.innerHTML = icon;
  button.style.cssText = 'display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;background:transparent;color:inherit;border:1px solid #8885;border-radius:6px;cursor:pointer;';
  const status = document.createElement('span');
  status.setAttribute('role', 'status');
  status.style.cssText = 'margin-left:8px;font-size:12px;';
  parentElement.replaceChildren(button, status);
  button.addEventListener('click', async () => {
    const text = String(data?.text ?? '');
    button.disabled = true;
    status.textContent = '';
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement('textarea');
        field.value = text;
        field.style.cssText = 'position:fixed;opacity:0;';
        parentElement.appendChild(field);
        try {
          field.select();
          if (!document.execCommand('copy')) throw new Error('Copy failed');
        } finally {
          field.remove();
          button.focus();
        }
      }
      button.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>';
      button.title = 'コピーしました';
      button.setAttribute('aria-label', 'コピーしました');
    } catch {
      status.textContent = 'コピーできませんでした。ブラウザーの権限をご確認ください。';
    } finally {
      button.disabled = false;
    }
  });
}
