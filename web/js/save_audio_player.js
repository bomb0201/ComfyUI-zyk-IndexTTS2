import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

const MARGIN = 6;
let stylesInjected = false;

function ensureStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
.zyk-indextts2-audio-player {
  position: absolute;
  z-index: 12;
  pointer-events: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px;
  background: var(--bg-color, var(--comfy-menu-bg, #2a2a2a));
  border: 1px solid var(--border-color, #444);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
  min-width: 180px;
  transition: opacity 0.2s ease, filter 0.2s ease;
}
.zyk-indextts2-audio-player[data-state="inactive"] {
  opacity: 0.9;
}
.zyk-indextts2-audio-player[data-state="inactive"] audio {
  pointer-events: none;
  filter: grayscale(0.8);
  opacity: 0.42;
}
.zyk-indextts2-audio-player__title {
  font-size: 11px;
  font-weight: 600;
  color: var(--descrip-text, #c7c7c7);
}
`;
  document.head.appendChild(style);
}

function buildTransformStyle(ctx, widgetWidth, y) {
  const { canvas } = ctx;
  const rect = canvas.getBoundingClientRect();
  const matrix = new DOMMatrix()
    .scaleSelf(rect.width / canvas.width, rect.height / canvas.height)
    .multiplySelf(ctx.getTransform())
    .translateSelf(MARGIN, y + MARGIN);
  return {
    transformOrigin: '0 0',
    transform: matrix.toString(),
    left: `${rect.left + window.scrollX}px`,
    top: `${rect.top + window.scrollY}px`,
  };
}

function buildAudioUrl(item) {
  if (!item || !item.filename) {
    return null;
  }

  const params = new URLSearchParams({
    filename: item.filename,
    type: item.type || 'output',
  });

  if (item.subfolder) {
    params.set('subfolder', item.subfolder);
  }

  let url = api.apiURL(`/view?${params.toString()}`);
  if (typeof app.getRandParam === 'function') {
    url += app.getRandParam();
  }
  return url;
}

function setState(container, audioEl, titleEl, clip) {
  if (clip?.url) {
    if (audioEl.src !== clip.url) {
      audioEl.src = clip.url;
      audioEl.load();
    }
    titleEl.textContent = clip.title ?? 'Audio preview';
    container.dataset.state = 'active';
  } else {
    audioEl.pause();
    audioEl.removeAttribute('src');
    audioEl.load();
    titleEl.textContent = 'No audio yet';
    container.dataset.state = 'inactive';
  }
}

app.registerExtension({
  name: 'zyk-IndexTTS2.SaveAudioPlayer',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'zyk-IndexTTS2SaveAudio') {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      originalOnNodeCreated?.apply(this, args);

      ensureStyles();

      const container = document.createElement('div');
      container.className = 'zyk-indextts2-audio-player';

      const title = document.createElement('div');
      title.className = 'zyk-indextts2-audio-player__title';
      title.textContent = 'No audio yet';

      const audio = document.createElement('audio');
      audio.controls = true;
      audio.style.width = '100%';

      container.appendChild(title);
      container.appendChild(audio);
      document.body.appendChild(container);

      const widget = {
        name: 'zyk_indextts2_audio_widget',
        type: 'zyk_indextts2_audio_widget',
        draw(ctx, node, widgetWidth, y) {
          if (!container.isConnected) {
            document.body.appendChild(container);
          }
          const baseWidth = Math.max(160, (node.size?.[0] ?? widgetWidth) - MARGIN * 2);
          container.style.width = `${baseWidth}px`;
          const style = buildTransformStyle(ctx, widgetWidth, y);
          Object.assign(container.style, style);
        },
        serialize: false,
        computeSize() {
          return [220, 90];
        },
      };

      this.addCustomWidget(widget);
      this.size = [this.size[0], Math.max(this.size[1], 120)];

      const originalOnRemoved = this.onRemoved;
      this.onRemoved = function () {
        container.remove();
        originalOnRemoved?.apply(this, arguments);
      };
      setState(container, audio, title, null);

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        const audioResults =
          message?.ui?.audio ||
          message?.audio ||
          [];

        if (Array.isArray(audioResults) && audioResults.length > 0) {
          const latest = audioResults[audioResults.length - 1];
          const url = buildAudioUrl(latest);
          if (url) {
            setState(container, audio, title, {
              url,
              title: latest.filename || 'Audio preview',
            });
            return;
          }
        }

        setState(container, audio, title, null);
      };
    };
  },
});
