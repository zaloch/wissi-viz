// NiceGUI (Vue) wrapper around the <wsi-grid> web component.
import { WsiGrid } from "wissi-viz";

const EVENTS = ["viewchange", "paneclick", "objectclick", "draw", "activepane", "slideload"];

export default {
  template: `<div style="position:relative"></div>`,
  props: {
    api: String,
    slides: Array,
    layout: String,
    sync: String,
    controls: Boolean,
    tool: String,
    drawLayer: String,
    quality: Number,
    textureBudgetMb: Number,
  },
  mounted() {
    void WsiGrid; // ensure the custom element is registered
    const el = document.createElement("wsi-grid");
    el.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
    el.setAttribute("api", (window.path_prefix || "") + this.api);
    if (this.layout) el.setAttribute("layout", this.layout);
    if (this.controls === false) el.setAttribute("controls", "false");
    if (this.quality) el.setAttribute("quality", String(this.quality));
    if (this.drawLayer) el.setAttribute("draw-layer", this.drawLayer);
    for (const ev of EVENTS) el.addEventListener(ev, (e) => this.$emit(ev, e.detail));
    this.$el.appendChild(el);
    this.grid = el;
    el.setSlides(this.slides || []);
    if (this.sync) el.setSync(this.sync);
    if (this.tool) el.setTool(this.tool);
    if (this.textureBudgetMb) el.setTextureBudget(this.textureBudgetMb);
  },
  unmounted() {
    this.grid?.fetcher?.destroy();
  },
  watch: {
    slides(v) { this.grid.setSlides(v || []); },
    layout(v) { this.grid.setLayout(v); },
    sync(v) { this.grid.setSync(v); },
    tool(v) { this.grid.setTool(v); },
    drawLayer(v) { this.grid.drawLayer = v || null; },
    quality(v) { this.grid.quality = v || 1; this.grid.requestRender(); },
  },
  methods: {
    call(name, ...args) {
      const fn = this.grid?.[name];
      if (typeof fn !== "function") throw new Error(`wsi-grid has no method ${name}`);
      return fn.apply(this.grid, args);
    },
  },
};
