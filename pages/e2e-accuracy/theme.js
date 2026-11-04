/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
"use strict";
(() => {
  const system = window.matchMedia("(prefers-color-scheme: dark)");
  let mode = "system";
  try { mode = localStorage.getItem("aisim-theme-mode") || "system"; } catch (_) {}
  if (!["system", "light", "dark"].includes(mode)) mode = "system";
  function apply() {
    const theme = mode === "system" ? (system.matches ? "dark" : "light") : mode;
    document.documentElement.dataset.theme = theme;
    // Existing site pages consume sm-theme; the preference remains separate from the resolved color.
    try { localStorage.setItem("sm-theme", theme); } catch (_) {}
    document.querySelectorAll(".replay-dashboard").forEach(root => root.classList.toggle("light", theme === "light"));
    window.dispatchEvent(new CustomEvent("aisim-theme", {detail:theme}));
  }
  apply();
  system.addEventListener("change", () => { if (mode === "system") apply(); });
  document.addEventListener("DOMContentLoaded", () => {
    const control = document.getElementById("theme-mode");
    if (!control) return;
    control.value = mode;
    control.addEventListener("change", () => {
      mode = control.value;
      try { localStorage.setItem("aisim-theme-mode", mode); } catch (_) {}
      apply();
    });
  });
})();
