document.addEventListener("DOMContentLoaded", () => {
    const currentPage = window.location.pathname.split("/").pop() || "index.html";

    document.querySelectorAll(".navigation a").forEach((link) => {
        const linkPage = link.getAttribute("href").replace("./", "");
        if (linkPage === currentPage) {
            link.setAttribute("aria-current", "page");
        }
    });
});
