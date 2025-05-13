// chat.js
document.addEventListener('DOMContentLoaded', function () {
  const API_URL = 'https://mi-api-flask-6i8o.onrender.com';

  // — Inicialización de Firebase —
  const firebaseConfig = {
    apiKey: "AIzaSyCp4C-DrKuLUxS9yo9VyBYa5CZxm1Q3NBI",
    authDomain: "ia-medica-6f09e.firebaseapp.com",
    projectId: "ia-medica-6f09e",
    storageBucket: "ia-medica-6f09e.appspot.com",
    messagingSenderId: "747957864751",
    appId: "1:747957864751:web:a09686be84ed0b3b5db9da"
  };
  firebase.initializeApp(firebaseConfig);
  const auth = firebase.auth();
  const db   = firebase.firestore();

  // — Selectores —
  const chatBox       = document.getElementById('chatBox');
  const toggleBtn     = document.getElementById('toggleHistory');
  const historyPanel  = document.getElementById('historyPanel');
  const newSessionBtn = document.getElementById('newSessionBtn');
  const sessionsList  = document.getElementById('sessionsList');
  const textInput     = document.getElementById('textInput');
  const imageInput    = document.getElementById('imageInput');
  const sendBtn       = document.getElementById('sendBtn');

  // — Helpers —
  function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
  }

  function appendMessage(type, text, dateObj) {
    const msgDate = dateObj || new Date();
    const bubble = document.createElement('div');
    bubble.className = `message ${type}`;
    bubble.innerHTML = `
      <div>${text}</div>
      <span class="timestamp">
        ${msgDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </span>
    `;
    if (type === 'ai') {
      const prompts = document.createElement('div');
      prompts.className = 'bubble-prompts';
      ['Resumen', 'Diagnóstico', 'Tratamientos'].forEach(label => {
        const btn = document.createElement('button');
        btn.textContent = label;
        btn.onclick = () => {
          textInput.value = label;
          sendBtn.click();
        };
        prompts.appendChild(btn);
      });
      bubble.appendChild(prompts);
    }
    chatBox.appendChild(bubble);
    scrollToBottom(chatBox);
  }

  async function saveMessage(type, text) {
    const sessionId = window.__currentSessionId;
    if (!sessionId || !window.currentUid) return;
    await db
      .collection('users').doc(window.currentUid)
      .collection('sessions').doc(sessionId)
      .collection('messages')
      .add({
        type,
        text,
        timestamp: firebase.firestore.FieldValue.serverTimestamp()
      });
  }

  // — Sesiones en Firestore —
  async function renderSessions() {
    if (!sessionsList || !window.currentUid) return;
    const snap = await db
      .collection('users').doc(window.currentUid)
      .collection('sessions')
      .orderBy('createdAt', 'desc')
      .get();
    sessionsList.innerHTML = '';
    snap.forEach(doc => {
      const data = doc.data();
      const item = document.createElement('div');
      item.className = 'session-item';
      item.textContent = data.title || 'Sin título';
      item.onclick = () => loadSession(doc.id);
      sessionsList.appendChild(item);
    });
  }

  async function createSession() {
    if (!window.currentUid) return;
    const ref = await db
      .collection('users').doc(window.currentUid)
      .collection('sessions')
      .add({
        title: 'Nueva conversación',
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });
    window.__currentSessionId = ref.id;
    await renderSessions();
    await loadSession(ref.id);
  }

  async function loadSession(sessionId) {
    if (!chatBox) return;
    window.__currentSessionId = sessionId;
    chatBox.innerHTML = '';
    const snap = await db
      .collection('users').doc(window.currentUid)
      .collection('sessions').doc(sessionId)
      .collection('messages')
      .orderBy('timestamp', 'asc')
      .get();
    snap.forEach(doc => {
      const { type, text, timestamp } = doc.data();
      appendMessage(type, text, timestamp && timestamp.toDate());
    });
  }

  async function generateSessionTitle() {
    const sessionId = window.__currentSessionId;
    if (!sessionId || !window.currentUid) return;
    const snap = await db
      .collection('users').doc(window.currentUid)
      .collection('sessions').doc(sessionId)
      .collection('messages')
      .orderBy('timestamp','asc')
      .get();
    const msgs = snap.docs.map(d => d.data().text).filter(Boolean);
    if (msgs.length < 2) return;
    try {
      const res = await fetch(`${API_URL}/api/generate-title`, {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({ messages: msgs })
      });
      if (res.ok) {
        const json = await res.json();
        const newTitle = json.title || 'Sin título';
        await db
          .collection('users').doc(window.currentUid)
          .collection('sessions').doc(sessionId)
          .update({ title: newTitle });
        await renderSessions();
      }
    } catch (e) {
      console.warn('Error generando título:', e);
    }
  }

  // — Envío de texto al chatbot —
  async function sendMessage() {
    const text = textInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    await saveMessage('user', text);

    const loader = document.createElement('div');
    loader.className = 'message ai typing-indicator';
    loader.innerHTML = `
      <div style="display:flex;align-items:center;">
        IA está escribiendo
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    `;
    chatBox.appendChild(loader);
    scrollToBottom(chatBox);

    let aiResponse = 'Lo siento, no recibí respuesta.';
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({
          session_id: window.__currentSessionId,
          message: text
        })
      });
      if (res.ok) {
        const data = await res.json();
        aiResponse = data.response || aiResponse;
      } else {
        console.error('Error /api/chat:', res.status, await res.text());
      }
    } catch (err) {
      console.error('Error comunicando con API chat:', err);
    }

    loader.remove();
    appendMessage('ai', aiResponse);
    await saveMessage('ai', aiResponse);
    await generateSessionTitle();

    textInput.value = '';
  }

  // — Subida y análisis automático de imagen —
  imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (ev) => {
      const dataURL = ev.target.result;
      appendMessage('user', `<img src="${dataURL}" style="max-width:200px;border-radius:8px;">`);
      await saveMessage('user', `<img src='${dataURL}'>`);

      const b64 = dataURL.split(',')[1];
      try {
        const resp = await fetch(`${API_URL}/api/analyze-image`, {
          method: 'POST',
          headers: { 'Content-Type':'application/json' },
          body: JSON.stringify({ image: b64 })
        });

        if (!resp.ok) {
          const txt = await resp.text();
          console.error('Analyze-image HTTP error:', resp.status, txt);
          appendMessage('ai', 'Ha ocurrido un error de red al analizar la imagen.');
          await saveMessage('ai', 'Ha ocurrido un error de red al analizar la imagen.');
        } else {
          const body = await resp.json();
          if (body.error) {
            appendMessage('ai', `Error al analizar la imagen: ${body.error}`);
            await saveMessage('ai', `Error al analizar la imagen: ${body.error}`);
          } else {
            const desc = (body.description || '').trim();
            // mostramos descripción o mensaje genérico
            appendMessage('ai', desc || 'Imagen recibida correctamente.');
            await saveMessage('ai', desc || 'Imagen recibida correctamente.');
            // pregunta de seguimiento
            appendMessage('ai', '¿Qué quieres que haga con esta información?');
            await saveMessage('ai', '¿Qué quieres que haga con esta información?');
          }
        }
      } catch (err) {
        console.error('Error al analizar la imagen:', err);
        appendMessage('ai', 'Ha ocurrido un error inesperado al analizar la imagen.');
        await saveMessage('ai', 'Ha ocurrido un error inesperado al analizar la imagen.');
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  });

  // — Botón enviar y Enter —
  sendBtn.addEventListener('click', sendMessage);
  textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // — Toggle historial en móvil —
  toggleBtn.addEventListener('click', () => {
    historyPanel.classList.toggle('open');
  });

  // — Autenticación y sesión inicial —
  auth.onAuthStateChanged(async user => {
    const loginUrl = '/iniciar-sesion';
    if (!user) {
      if (!location.pathname.includes(loginUrl)) location.href = loginUrl;
      return;
    }
    window.currentUid = user.uid;
    await renderSessions();
    if (!window.__currentSessionId) {
      await createSession();
    } else {
      await loadSession(window.__currentSessionId);
    }
    newSessionBtn.addEventListener('click', createSession);
  });
});
