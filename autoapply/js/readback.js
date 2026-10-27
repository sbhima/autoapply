// Read back the value currently shown by a field (after filling) so the review
// table reflects what is ACTUALLY on the page, not what we intended.
(args) => {
  const { uid, kind, optionUids } = args;
  const raw = (s) => (s || '').replace(/[\u00a0\s]+/g, ' ').trim();
  const find = (id) => {
    const sel = `[data-aa-id="${id}"]`;
    const stack = [document];
    while (stack.length) {
      const r = stack.pop();
      const hit = r.querySelector(sel);
      if (hit) return hit;
      for (const el of r.querySelectorAll('*')) if (el.shadowRoot) stack.push(el.shadowRoot);
    }
    return null;
  };
  const labelText = (el) => {
    const l = (el.labels && el.labels[0]) || el.closest('label') || document.querySelector(`[data-aa-label="${el.getAttribute('data-aa-id')}"]`);
    return raw(l ? l.textContent : (el.getAttribute('aria-label') || el.value || el.textContent));
  };
  if (kind === 'radio' || kind === 'checkbox_group') {
    const out = [];
    for (const id of optionUids || []) {
      const el = find(id); if (!el) continue;
      const on = el.checked || el.getAttribute('aria-checked') === 'true';
      if (on) out.push(labelText(el));
    }
    return out.join(', ');
  }
  if (kind === 'button_group') {
    for (const id of optionUids || []) {
      const el = find(id); if (!el) continue;
      const cls = (el.className && typeof el.className === 'string') ? el.className : '';
      if (el.getAttribute('aria-pressed') === 'true' || el.getAttribute('aria-checked') === 'true' || el.getAttribute('data-state') === 'on' || /(^|[_\s-])(active|selected|checked|isSelected)/i.test(cls)) return raw(el.textContent);
    }
    return '';
  }
  const el = find(uid);
  if (!el) return '__GONE__';
  if (kind === 'checkbox') return (el.checked || el.getAttribute('aria-checked') === 'true') ? 'true' : '';
  if (kind === 'select') {
    const so = el.selectedOptions && el.selectedOptions[0];
    return so && so.value !== '' ? raw(so.text) : '';
  }
  if (kind === 'file') {
    if (el.files && el.files.length) return [...el.files].map((f) => f.name).join(', ');
    // many ATSs swap the input for a "filename.pdf ✕" chip after upload
    let p = el.parentElement;
    for (let i = 0; i < 5 && p; i++, p = p.parentElement) {
      const m = raw(p.textContent).match(/[\w.\- ()]+\.(pdf|docx?|txt|rtf)/i);
      if (m) return m[0].trim();
    }
    return '';
  }
  if (kind === 'listbox_button') {
    const t = raw(el.textContent);
    return /^(select one|select|choose|--)/i.test(t) ? '' : t;
  }
  if (kind === 'combobox') {
    // react-select / Ashby / Workday: the chosen value is rendered next to the input
    const containers = [el.closest('[class*="select__control"]'), el.closest('[class*="-control"]'), el.closest('[class*="container"]'), el.closest('[data-automation-id="multiselectInputContainer"]'), el.parentElement && el.parentElement.parentElement];
    for (const c of containers) {
      if (!c) continue;
      const chips = c.querySelectorAll('[class*="single-value"], [class*="singleValue"], [class*="multi-value__label"], [class*="multiValue"] [class*="label"], [data-automation-id="selectedItem"], [data-automation-id="promptOption"]');
      if (chips.length) return [...chips].map((x) => raw(x.textContent)).filter(Boolean).join(', ');
    }
    if (el.value) return el.value;
    const v = el.getAttribute('aria-valuetext') || el.getAttribute('data-value');
    if (v) return raw(v);
    return '';
  }
  if (el.getAttribute('contenteditable') === 'true') return raw(el.textContent);
  return el.value || '';
}
