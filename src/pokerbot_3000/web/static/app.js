const statusItems = document.querySelectorAll("[data-status-item]");

for (const item of statusItems) {
  item.classList.add("is-ready");
}
