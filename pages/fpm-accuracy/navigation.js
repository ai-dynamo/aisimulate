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
  function setRun(snapshot) {
    const link = document.getElementById('evaluation-run');
    link.hidden = true;
    link.removeAttribute('href');
    if (snapshot && /^[1-9][0-9]*$/.test(snapshot.run_id) && /^[1-9][0-9]*$/.test(snapshot.run_attempt)) {
      link.href = `https://github.com/ai-dynamo/aisimulate/actions/runs/${snapshot.run_id}/attempts/${snapshot.run_attempt}`;
      link.hidden = false;
    }
  }
  window.fpmNavigation = {setRun};
  if (page === '3d-visualization.html') {
    document.getElementById('evaluation-run').textContent = 'Latest evaluation run';
    (async () => {
      const response = await fetch('branches.json', {cache:'no-cache'});
      if (!response.ok) return;
      const catalog = await response.json();
      const selected = new URLSearchParams(location.search).get('branch') || catalog.default_branch;
      const entry = catalog.branches.find(item => item.branch === selected);
      if (entry?.status !== 'available' || !/^branches\/[0-9a-f]{16}\/summary\.json$/.test(entry.summary_path)) return;
      const result = await fetch(entry.summary_path, {cache:'no-cache'});
      if (!result.ok) return;
      const summary = await result.json();
      if (summary.snapshot?.branch === selected) setRun(summary.snapshot);
    })().catch(() => setRun(null));
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
