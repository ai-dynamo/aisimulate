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
  if (page !== 'index.html') {
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
