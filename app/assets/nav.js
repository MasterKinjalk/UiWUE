/* UiWUE shell behaviour (app/app.py): the current page in the nav, and readable hover tips on every page. */
(function () {
  "use strict";

  /* ---- nav: class "active" and aria-current on the link of the current page, its name on the phone menu
     button, and the menu folded after a pick (a dcc.Link navigates without a reload, so the menu stays open). */
  window.dash_clientside = Object.assign({}, window.dash_clientside);
  window.dash_clientside.uiwue = Object.assign({}, window.dash_clientside.uiwue, {
    markPage: function (path) {
      var name = null;
      document.querySelectorAll(".uw-nav a[href]").forEach(function (a) {
        var on = a.getAttribute("href").split("?")[0] === path;
        a.classList.toggle("active", on);
        if (on) {
          a.setAttribute("aria-current", "page");
          if (!a.classList.contains("uw-brand")) name = a.textContent;
        } else {
          a.removeAttribute("aria-current");
        }
      });
      var menu = document.getElementById("uw-menu");
      if (menu) menu.removeAttribute("open");
      return name || (path === "/" ? "Home" : "Pages");
    },
  });

  /* ---- the selection travels: a nav link takes ?bed=&plant=&day= from the current URL to a page whose layout reads
     them (window.UW_CARRY, from app.py). The href is set when the link is pointed at or focused, so a new tab gets
     it too; a plain click navigates as dcc.Link does (pushState and Dash's own event), with the carried query. */
  function carriedHref(a) {
    var path = a.getAttribute("href").split("?")[0];
    var keys = (window.UW_CARRY || {})[path];
    if (!keys || path === window.location.pathname) return path;
    var here = new URLSearchParams(window.location.search);
    var out = new URLSearchParams();
    keys.forEach(function (k) {
      var v = here.get(k);
      if (v && (k !== "plant" || here.get("bed"))) out.set(k, v);
    });
    var q = out.toString();
    return q ? path + "?" + q : path;
  }

  function refresh(e) {
    var a = e.target.closest && e.target.closest(".uw-nav a.nav-link");
    if (a) a.setAttribute("href", carriedHref(a));
  }
  document.addEventListener("mouseover", refresh);
  document.addEventListener("focusin", refresh);

  document.addEventListener(
    "click",
    function (e) {
      var a = e.target.closest && e.target.closest(".uw-nav a.nav-link");
      if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      var href = carriedHref(a);
      if (href.indexOf("?") < 0) return; // nothing to carry: dcc.Link navigates as usual
      e.preventDefault();
      e.stopPropagation();
      a.setAttribute("href", href);
      window.history.pushState({}, "", href);
      window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
      window.scrollTo(0, 0);
    },
    true
  );

  document.addEventListener("keydown", function (e) {
    var menu = document.getElementById("uw-menu");
    if (e.key === "Escape" && menu && menu.open) {
      menu.removeAttribute("open");
      menu.querySelector("summary").focus();
    }
  });

  /* ---- hover tips: an element's title shows in a styled box (DESIGN: surface, gray-500 edge, sans 13, at most
     320 px wide, so a long method note wraps into lines a person can read). It also shows on keyboard focus and
     on a tap of a non-link element (a chip), which a native title never does. While the tip shows, the title moves
     to data-uw-title (the tip is tied to the element with aria-describedby), and it comes back when the tip goes. */
  var tip = null;
  var owner = null;
  var timer = null;

  function box() {
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "uw-tip";
      tip.className = "uw-tip";
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    return tip;
  }

  function target(node) {
    while (node && node !== document.body) {
      if (node.nodeType === 1 && (node.hasAttribute("title") || node.hasAttribute("data-uw-title"))) return node;
      node = node.parentNode;
    }
    return null;
  }

  function hide() {
    clearTimeout(timer);
    if (owner) {
      var text = owner.getAttribute("data-uw-title");
      owner.removeAttribute("data-uw-title");
      if (text !== null && !owner.hasAttribute("title")) owner.setAttribute("title", text);
      owner.removeAttribute("aria-describedby");
      owner = null;
    }
    if (tip) tip.classList.remove("uw-tip-on");
  }

  function place(el) {
    var r = el.getBoundingClientRect();
    var t = box();
    var w = t.offsetWidth;
    var h = t.offsetHeight;
    var gap = 8;
    var x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    var y = r.bottom + gap;
    if (y + h > window.innerHeight - 8 && r.top - gap - h > 8) y = r.top - gap - h;
    t.style.left = Math.round(x + window.scrollX) + "px";
    t.style.top = Math.round(y + window.scrollY) + "px";
  }

  function show(el) {
    if (owner === el) return;
    hide();
    var text = el.getAttribute("title");
    if (!text) return;
    owner = el;
    el.setAttribute("data-uw-title", text);
    el.removeAttribute("title");
    var t = box();
    t.textContent = text;
    el.setAttribute("aria-describedby", "uw-tip");
    t.classList.add("uw-tip-on");
    place(el);
  }

  document.addEventListener("mouseover", function (e) {
    var el = target(e.target);
    if (el === owner) return;
    clearTimeout(timer);
    if (!el) {
      hide();
      return;
    }
    if (!el.hasAttribute("title")) return;
    // the native tooltip must not show alongside: take the title at once, show the box after a short rest
    var text = el.getAttribute("title");
    hide();
    owner = el;
    el.setAttribute("data-uw-title", text);
    el.removeAttribute("title");
    timer = setTimeout(function () {
      if (owner !== el) return;
      var t = box();
      t.textContent = text;
      el.setAttribute("aria-describedby", "uw-tip");
      t.classList.add("uw-tip-on");
      place(el);
    }, 250);
  });

  document.addEventListener("focusin", function (e) {
    var el = target(e.target);
    if (el && el === e.target && el.hasAttribute("title") && el.matches(":focus-visible")) show(el);
  });
  document.addEventListener("focusout", function (e) {
    if (owner && owner === e.target) hide();
  });

  document.addEventListener("click", function (e) {
    var el = target(e.target);
    var interactive = e.target.closest && e.target.closest("a, button, input, select, label, summary, [role=button]");
    if (el && !interactive && el !== owner && el.hasAttribute("title")) {
      show(el);
    } else if (!el || el !== owner) {
      hide();
    }
  });

  document.addEventListener("mouseout", function (e) {
    if (!e.relatedTarget) hide(); // the pointer left the window
  });
  window.addEventListener("scroll", hide, { passive: true });
  window.addEventListener("resize", hide);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") hide();
  });
})();
