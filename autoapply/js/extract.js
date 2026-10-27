// autoapply DOM scanner. Evaluated inside every frame via Playwright.
// Returns a list of logical form fields (radio groups collapsed, labels resolved),
// tagging each element with a stable data-aa-id so Python can locate it later.
// Pierces open shadow roots (SmartRecruiters, some Workday widgets).
(opts) => {
  const PFX = (opts && opts.prefix) || 'f0';
  const AA = 'data-aa-id';
  let counter = Number(document.documentElement.getAttribute('data-aa-counter') || '0');
  const nextId = () => { counter += 1; return `${PFX}-${counter}`; };
  const uidOf = (el) => {
    let id = el.getAttribute(AA);
    if (!id) { id = nextId(); el.setAttribute(AA, id); }
    return id;
  };

  const raw = (s) => (s || '').replace(/[\u00a0\s]+/g, ' ').trim();
  const cleanLabel = (s) => raw(s)
    .replace(/\s*\((required|optional)\)\s*/ig, ' ')
    .replace(/\s*[*✱]+\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  // ---------------------------------------------------------------- traversal
  const roots = [document];
  (function collectShadow(root) {
    const all = root.querySelectorAll('*');
    for (const el of all) {
      if (el.shadowRoot) { roots.push(el.shadowRoot); collectShadow(el.shadowRoot); }
    }
  })(document);
  const qsa = (sel) => { const out = []; for (const r of roots) out.push(...r.querySelectorAll(sel)); return out; };
  const byId = (el, id) => {
    const rn = el.getRootNode();
    return (rn && rn.getElementById) ? rn.getElementById(id) : document.getElementById(id);
  };

  // ---------------------------------------------------------------- visibility
  const isShown = (el) => {
    if (!el || !el.isConnected) return false;
    if (el.checkVisibility) {
      if (!el.checkVisibility({ checkVisibilityCSS: true })) return false;
    } else {
      const st = getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden') return false;
    }
    const r = el.getBoundingClientRect();
    if (r.width < 1 && r.height < 1) return false;
    if (r.right < -500 || r.bottom < -500) return false; // off-screen honeypots
    return true;
  };
  const labelsOf = (el) => {
    const out = [];
    try { if (el.labels) out.push(...el.labels); } catch (e) {}
    if (el.id) {
      for (const r of roots) {
        try { out.push(...r.querySelectorAll(`label[for="${CSS.escape(el.id)}"]`)); } catch (e) {}
      }
    }
    const wrap = el.closest('label'); if (wrap) out.push(wrap);
    return [...new Set(out)];
  };
  // radios/checkboxes are often visually hidden with a styled label instead
  const isShownToggle = (el) => isShown(el) || labelsOf(el).some(isShown) || (el.parentElement && isShown(el.parentElement) && getComputedStyle(el).opacity === '0');

  const EXCLUDE_ANCESTOR = 'nav, header:not(form header), footer:not(form footer), [role="search"], [role="navigation"], #onetrust-consent-sdk, .cookie, [class*="cookie-banner"], [id*="cookie"], [aria-modal="true"][class*="cookie"]';
  const isExcluded = (el) => {
    if (el.closest(EXCLUDE_ANCESTOR) && !el.closest('form')) return true;
    const sig = `${el.name || ''} ${el.id || ''} ${el.className && el.className.baseVal === undefined ? el.className : ''}`.toLowerCase();
    if (/(g-recaptcha|h-captcha|hcaptcha|captcha-response|honeypot|hp_field|search_query|searchbox-nav)/.test(sig)) return true;
    if (el.getAttribute('tabindex') === '-1' && el.tagName === 'INPUT' && el.type !== 'file' && el.type !== 'radio' && el.type !== 'checkbox' && el.getAttribute('role') !== 'combobox' && !isShown(el)) return true;
    return false;
  };

  // ---------------------------------------------------------------- text helpers
  const CTRL_SEL = 'input:not([type="hidden"]):not([type="submit"]):not([type="button"]), select, textarea, [role="combobox"], [role="radio"], [role="checkbox"], button[aria-haspopup="listbox"]';
  const textOf = (node) => {
    if (!node) return '';
    const clone = node.cloneNode(true);
    clone.querySelectorAll('input, select, textarea, option, script, style, svg, noscript, [role="listbox"], [role="option"], [aria-hidden="true"], .visually-hidden.select__a11y, [class*="a11yText"]').forEach((x) => x.remove());
    // insert spaces between block children so text doesn't run together
    clone.querySelectorAll('div, p, li, br, span, label, legend').forEach((x) => { x.append(' '); });
    return raw(clone.textContent);
  };
  const textByIds = (el, attr) => {
    const ids = (el.getAttribute(attr) || '').split(/\s+/).filter(Boolean);
    return raw(ids.map((id) => { const t = byId(el, id); return t ? textOf(t) : ''; }).filter(Boolean).join(' '));
  };
  const ctrlCount = (node, groupName) => {
    let n = 0; const seen = new Set();
    for (const c of node.querySelectorAll(CTRL_SEL)) {
      if (c.type === 'radio' || c.getAttribute('role') === 'radio') {
        const key = 'r:' + (c.name || (c.closest('[role="radiogroup"]') || {}).id || Math.random());
        if (groupName && key === 'r:' + groupName) continue;
        if (seen.has(key)) continue; seen.add(key);
      }
      // inner input of an ARIA 1.1 combobox wrapper counts once
      if (c.matches('input') && c.closest('[role="combobox"]') && c.closest('[role="combobox"]') !== c) continue;
      n++;
    }
    return n;
  };
  // Text located immediately before `node` (siblings, then ancestors' siblings),
  // stopping once we would cross into a region containing other controls.
  const precedingText = (node, allowed) => {
    let cur = node;
    for (let depth = 0; depth < 7 && cur && cur.parentElement; depth++) {
      let sib = cur.previousElementSibling; let k = 0;
      while (sib && k < 4) {
        if (ctrlCount(sib) === 0) {
          const t = textOf(sib);
          if (t && t.length < 500) return t;
        } else break;
        sib = sib.previousElementSibling; k++;
      }
      const parent = cur.parentElement;
      if (ctrlCount(parent) > allowed) break;
      if (parent.tagName === 'LABEL' || parent.tagName === 'LEGEND') return textOf(parent);
      cur = parent;
    }
    return '';
  };
  const labelFor = (el) => {
    let t = textByIds(el, 'aria-labelledby');
    if (t) return t;
    const ls = labelsOf(el);
    if (ls.length) { t = raw(ls.map(textOf).filter(Boolean).join(' ')); if (t) return t; }
    t = raw(el.getAttribute('aria-label'));
    if (t && !/^(select|choose|search)\.*$/i.test(t)) return t;
    t = precedingText(el, 1);
    if (t) return t;
    return raw(el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || el.id || '');
  };
  const describe = (el) => {
    const t = textByIds(el, 'aria-describedby');
    return t.length < 600 ? t : t.slice(0, 600);
  };
  const lca = (els) => {
    if (!els.length) return null;
    let a = els[0].parentElement;
    while (a && !els.every((e) => a.contains(e))) a = a.parentElement;
    return a;
  };
  const groupLabel = (container, members, groupKey) => {
    if (!container) return '';
    // fieldset legend / radiogroup labelling
    const fs = container.closest('fieldset');
    if (fs) { const lg = fs.querySelector('legend'); if (lg) { const t = textOf(lg); if (t) return t; } }
    let rg = container.closest('[role="radiogroup"], [role="group"]');
    if (rg) {
      const t = textByIds(rg, 'aria-labelledby') || raw(rg.getAttribute('aria-label'));
      if (t) return t;
    }
    // text inside the container that is not part of an option label
    const optLabelNodes = new Set();
    members.forEach((m) => labelsOf(m).forEach((l) => optLabelNodes.add(l)));
    const clone = container.cloneNode(true);
    // remove option labels by matching text
    const optTexts = new Set(members.map((m) => raw(labelsOf(m).map(textOf).join(' '))).filter(Boolean));
    clone.querySelectorAll('label').forEach((l) => { if (optTexts.has(textOf(l))) l.remove(); });
    let t = textOf(clone);
    for (const ot of optTexts) t = t.replace(ot, '');
    t = raw(t);
    if (t && t.length < 500) return t;
    return precedingText(container, 1);
  };
  const isRequired = (el, labelRaw) => {
    if (el.required || el.getAttribute('aria-required') === 'true') return true;
    if (/[*✱]\s*$|^\s*[*✱]|\*\s*\)|\(required\)/i.test(labelRaw)) return true;
    return false;
  };
  const sectionOf = (el) => {
    const fs = el.closest('fieldset');
    if (fs) { const lg = fs.querySelector('legend'); if (lg) return textOf(lg).slice(0, 80); }
    let cur = el.parentElement;
    for (let d = 0; d < 8 && cur && cur !== document.body; d++, cur = cur.parentElement) {
      const h = cur.querySelector(':scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > header h2, :scope > header h3, :scope > [data-automation-id*="Header"]');
      if (h) {
        const t = textOf(h);
        if (t && t.length < 90) return t;
      }
    }
    return '';
  };

  // ---------------------------------------------------------------- collection
  const fields = [];
  const handled = new Set();

  // 1) native radios grouped by name (+ ARIA radios grouped by radiogroup)
  const radioGroups = new Map();
  for (const r of qsa('input[type="radio"], [role="radio"]')) {
    if (isExcluded(r) || !isShownToggle(r)) continue;
    let key;
    if (r.tagName === 'INPUT') key = 'n:' + (r.name || r.id);
    else { const g = r.closest('[role="radiogroup"]'); key = 'g:' + (g ? uidOf(g) : uidOf(r.parentElement)); }
    if (!radioGroups.has(key)) radioGroups.set(key, []);
    radioGroups.get(key).push(r);
  }
  for (const [key, members] of radioGroups) {
    const container = members.length > 1 ? lca(members) : (members[0].closest('fieldset, [role="radiogroup"]') || members[0].parentElement.parentElement);
    const labelRaw = groupLabel(container, members, key);
    const options = members.map((m) => {
      const lt = raw(labelsOf(m).map(textOf).join(' ')) || raw(m.getAttribute('aria-label')) || textOf(m) || m.value;
      // tag the label too so python can click it if the input is hidden
      const lab = labelsOf(m)[0]; const uid = uidOf(m); if (lab) lab.setAttribute('data-aa-label', uid);
      return { text: cleanLabel(lt), value: m.value || '', uid };
    });
    members.forEach((m) => handled.add(m));
    const req = members.some((m) => m.required || m.getAttribute('aria-required') === 'true') || isRequired(container || members[0], labelRaw) || (container && container.closest('[role="radiogroup"][aria-required="true"]'));
    fields.push({ uid: 'grp-' + options[0].uid, kind: 'radio', label: cleanLabel(labelRaw), labelRaw, name: members[0].name || '', required: !!req, options, section: sectionOf(members[0]) });
  }

  // 2) checkboxes: group when several share a name or sit in one fieldset/group
  const cbs = qsa('input[type="checkbox"], [role="checkbox"]').filter((c) => !isExcluded(c) && isShownToggle(c) && !c.closest('[role="option"], [role="listbox"]'));
  const cbGroups = new Map();
  for (const c of cbs) {
    let key = c.name && c.name.replace(/\[\]$/, '');
    const fs = c.closest('fieldset, [role="group"]');
    if (fs && fs.querySelectorAll('input[type="checkbox"], [role="checkbox"]').length > 1) key = 'fs:' + uidOf(fs);
    else if (!key || qsa(`input[type="checkbox"][name="${CSS.escape(c.name)}"]`).length < 2) key = 'single:' + uidOf(c);
    if (!cbGroups.has(key)) cbGroups.set(key, []);
    cbGroups.get(key).push(c);
  }
  for (const [key, members] of cbGroups) {
    members.forEach((m) => handled.add(m));
    if (members.length === 1) {
      const c = members[0];
      const uid = uidOf(c);
      const lab = labelsOf(c)[0]; if (lab) lab.setAttribute('data-aa-label', uid);
      let labelRaw = labelFor(c);
      const own = raw(labelsOf(c).map(textOf).join(' '));
      // a lone checkbox labelled just "Yes"/"I agree": prepend the question text
      if (own && own.length < 40) { const q = precedingText(c.closest('label') || c, 1); if (q && q !== own) labelRaw = q + ' — ' + own; }
      fields.push({ uid, kind: 'checkbox', label: cleanLabel(labelRaw), labelRaw, name: c.name || '', required: isRequired(c, labelRaw), options: [], current: (c.checked || c.getAttribute('aria-checked') === 'true') ? 'true' : '', description: describe(c), section: sectionOf(c) });
    } else {
      const container = lca(members);
      const labelRaw = groupLabel(container, members, key);
      const options = members.map((m) => {
        const uid = uidOf(m); const lab = labelsOf(m)[0]; if (lab) lab.setAttribute('data-aa-label', uid);
        return { text: cleanLabel(raw(labelsOf(m).map(textOf).join(' ')) || raw(m.getAttribute('aria-label')) || m.value), value: m.value || '', uid };
      });
      fields.push({ uid: 'grp-' + options[0].uid, kind: 'checkbox_group', multi: true, label: cleanLabel(labelRaw), labelRaw, name: members[0].name || '', required: isRequired(container || members[0], labelRaw), options, section: sectionOf(members[0]) });
    }
  }

  // 3) yes/no style button groups (Ashby & friends)
  const btnParents = new Set();
  for (const b of qsa('button, [role="button"]')) {
    if (!isShown(b) || isExcluded(b)) continue;
    const p = b.parentElement; if (!p || btnParents.has(p)) continue;
    const btns = [...p.children].filter((x) => x.matches('button, [role="button"]'));
    if (btns.length < 2 || btns.length > 5 || [...p.children].length !== btns.length) continue;
    const texts = btns.map((x) => raw(x.textContent));
    if (!texts.every((t) => t && t.length <= 30)) continue;
    if (!texts.some((t) => /^(yes|no)$/i.test(t))) continue;
    btnParents.add(p);
    const labelRaw = precedingText(p, 0) || textByIds(p, 'aria-labelledby');
    const options = btns.map((x) => ({ text: raw(x.textContent), value: '', uid: uidOf(x) }));
    fields.push({ uid: 'grp-' + options[0].uid, kind: 'button_group', label: cleanLabel(labelRaw), labelRaw, required: isRequired(p, labelRaw), options, section: sectionOf(p) });
  }

  // 4) everything else
  const candidates = qsa('input, textarea, select, [role="combobox"], button[aria-haspopup="listbox"], [contenteditable="true"][role="textbox"]');
  for (const el of candidates) {
    if (handled.has(el)) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && ['hidden', 'submit', 'button', 'reset', 'image', 'radio', 'checkbox'].includes(type)) continue;
    if (isExcluded(el)) continue;
    if (el.disabled && type !== 'file') continue;
    if (el.readOnly && el.getAttribute('role') !== 'combobox' && type !== 'file' && !el.closest('[role="combobox"]')) continue;
    // ARIA 1.1 combobox wrapper with an inner textbox: keep only the inner input
    if (el.getAttribute('role') === 'combobox' && tag !== 'input' && el.querySelector('input:not([type="hidden"])')) continue;
    if (el.closest('[role="listbox"], [role="option"]')) continue;

    let kind;
    if (type === 'file') kind = 'file';
    else if (tag === 'select') kind = 'select';
    else if (tag === 'textarea') kind = 'textarea';
    else if (tag === 'button') kind = 'listbox_button';
    else if (el.getAttribute('role') === 'combobox' || el.closest('[role="combobox"]') || el.getAttribute('aria-autocomplete') === 'list') kind = 'combobox';
    else if (type === 'date' || type === 'month') kind = 'date';
    else if (el.getAttribute('contenteditable') === 'true') kind = 'textarea';
    else kind = 'text';

    let shown = isShown(el);
    if (!shown) {
      if (kind === 'file') shown = true; // file inputs are hidden behind "Attach" buttons
      else if (kind === 'select' && (el.classList.contains('select2-hidden-accessible') || (el.nextElementSibling && /select2|chosen|selectize/.test(el.nextElementSibling.className || '')))) shown = true;
      else if (kind === 'combobox' && el.parentElement && isShown(el.parentElement)) shown = true; // react-select input is 1-2px wide
    }
    if (!shown) continue;

    const uid = uidOf(el);
    let labelRaw = labelFor(el);
    if (kind === 'listbox_button') {
      // Workday style: button text is the current value; label comes from the field container
      const own = raw(el.textContent);
      if (labelRaw === own || !labelRaw) labelRaw = precedingText(el, 1) || raw(el.getAttribute('aria-label'));
    }
    const f = {
      uid, kind, itype: type || (tag === 'textarea' ? 'textarea' : ''), label: cleanLabel(labelRaw), labelRaw,
      name: el.getAttribute('name') || '', id: el.id || '', required: isRequired(el, labelRaw),
      placeholder: raw(el.getAttribute('placeholder')), description: describe(el), section: sectionOf(el),
      autocomplete: el.getAttribute('autocomplete') || '', options: [], multi: !!el.multiple,
      current: '',
    };
    if (kind === 'select') {
      f.options = [...el.options].filter((o) => !(o.value === '' && /^(\s*|select.*|choose.*|please.*|--.*|none selected)$/i.test(raw(o.text)))).map((o) => ({ text: raw(o.text), value: o.value }));
      const so = el.selectedOptions && el.selectedOptions[0];
      f.current = so && so.value !== '' ? raw(so.text) : '';
    } else if (kind === 'file') {
      f.current = el.files && el.files.length ? [...el.files].map((x) => x.name).join(', ') : '';
      f.accept = el.getAttribute('accept') || '';
    } else if (kind === 'listbox_button') {
      const t = raw(el.textContent);
      f.current = /^(select one|select|choose|--)/i.test(t) ? '' : t;
    } else if (kind === 'combobox') {
      f.current = '';
    } else {
      f.current = el.value || (el.getAttribute('contenteditable') ? raw(el.textContent) : '');
    }
    fields.push(f);
  }

  document.documentElement.setAttribute('data-aa-counter', String(counter));
  return fields;
}
