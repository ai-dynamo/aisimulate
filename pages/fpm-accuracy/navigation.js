/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
(() => {
  const page = location.pathname.split('/').pop() || 'index.html';
  const branch = document.getElementById('branch');
  function links() {
    const value = new URLSearchParams(location.search).get('branch') || 'main';
    document.querySelectorAll('.fpm-tabs a').forEach(a => {
      const target = a.getAttribute('href').split('?')[0];
      a.href = target + '?branch=' + encodeURIComponent(value);
      if (target === page) a.setAttribute('aria-current', 'page');
    });
  }
  links();
  branch?.addEventListener('change', () => queueMicrotask(links));
  function setSnapshot(snapshot, message = 'No completed evaluation') {
    const banner = document.getElementById('evaluation-banner');
    banner.replaceChildren();
    if (!snapshot || !/^[1-9][0-9]*$/.test(snapshot.run_id) || !/^[1-9][0-9]*$/.test(snapshot.run_attempt)) {
      banner.textContent = message;
      return;
    }
    const item = (tag, text, href, id) => {
      const element = document.createElement(tag);
      element.textContent = text;
      if (href) { element.href = href; element.target = '_blank'; element.rel = 'noopener'; }
      if (id) element.id = id;
      return element;
    };
    const date = new Date(snapshot.completed_at).toISOString().slice(0,10);
    const time = item('time', date);
    time.dateTime = date;
    const items = [
      item('span', 'Daily evaluation'), time,
      item('a', 'Evaluation run', `https://github.com/ai-dynamo/aisimulate/actions/runs/${snapshot.run_id}/attempts/${snapshot.run_attempt}`, 'evaluation-run'),
      item('a', `AISim ${snapshot.commit_sha.slice(0,8)}`, `https://github.com/ai-dynamo/aisimulate/commit/${encodeURIComponent(snapshot.commit_sha)}`, 'evaluation-aisim'),
      item('a', `HF ${snapshot.hf_revision.slice(0,8)}`, `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/${encodeURIComponent(snapshot.hf_revision)}`, 'evaluation-hf'),
    ];
    items.forEach((element, index) => {
      if (index) {
        const separator = item('span', '·');
        separator.setAttribute('aria-hidden', 'true');
        banner.append(separator);
      }
      banner.append(element);
    });
  }
  window.fpmNavigation = {setSnapshot};
  if (page === '3d-visualization.html') {
    (async () => {
      setSnapshot(null, 'Loading evaluation…');
      const response = await fetch('branches.json', {cache:'no-cache'});
      if (!response.ok) { setSnapshot(null); return; }
      const catalog = await response.json();
      const selected = new URLSearchParams(location.search).get('branch') || catalog.default_branch;
      const entry = catalog.branches.find(item => item.branch === selected);
      if (entry?.status !== 'available' || !/^branches\/[0-9a-f]{16}\/summary\.json$/.test(entry.summary_path)) { setSnapshot(null); return; }
      const result = await fetch(entry.summary_path, {cache:'no-cache'});
      if (!result.ok) { setSnapshot(null); return; }
      const summary = await result.json();
      setSnapshot(summary.snapshot?.branch === selected ? summary.snapshot : null);
    })().catch(() => setSnapshot(null));
  }
  {
    const button = document.getElementById('theme-toggle');
    const label = () => button.setAttribute('aria-label', 'Switch to ' + (document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark') + ' theme');
    button.addEventListener('click', () => {
      const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = theme;
      try { localStorage.setItem('sm-theme', theme); } catch (_) { /* Optional persistence. */ }
      label();
    });
    label();
  }
})();
