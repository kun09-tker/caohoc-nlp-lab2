const messagesEl = document.getElementById('messages');
const input = document.getElementById('msgInput');
const sendBtn = document.getElementById('sendBtn');
const statusDot = document.querySelector('.status .dot');
const statusText = document.querySelector('.status');
const composer = document.getElementById('composer');
const ragSwitch = document.getElementById('ragSwitch');
const modeLabel = document.getElementById('modeLabel');
// let backendReadyFlag = false;
// let backendCheckInterval = null;


ragSwitch.addEventListener('change', () => {
  updateModeColors();
});

function updateModeColors() {
  if (ragSwitch.checked) {
    document.body.classList.add('rag-mode');
    document.body.classList.remove('qa-mode');
    modeLabel.textContent = 'RAG mode';
  } else {
    document.body.classList.add('qa-mode');
    document.body.classList.remove('rag-mode');
    modeLabel.textContent = 'QA mode';
  }
}

// Tạo một tin nhắn
function addMessage({ who='you', text='', meta='' }) {
  const row = document.createElement('div');
  row.className = 'row ' + (who === 'me' ? 'me' : 'you');

  if (who === 'you') {
    const avatarEl = document.createElement('div');
    avatarEl.className = 'avatar-small';
    avatarEl.innerHTML = `
        <img
            src="../../images/science_icon.png"
            alt="Avatar"
            style="
            width:100%;
            height:100%;
            border-radius:50%;
            object-fit:cover;
            "
        />
        `;
    row.appendChild(avatarEl);
  }

  const bubbleWrap = document.createElement('div');
  bubbleWrap.style.display = 'flex';
  bubbleWrap.style.flexDirection = 'column';

  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (who === 'me' ? 'me' : 'you');
  if (who === 'you') {
    bubble.dataset.mode = ragSwitch.checked ? 'rag' : 'qa';
  }
  bubble.textContent = text;
  bubbleWrap.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'meta';
    metaEl.textContent = meta;
    bubbleWrap.appendChild(metaEl);
  }

  row.appendChild(bubbleWrap);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Thêm ví dụ ban đầu
addMessage({ who:'you', text:'Xin chào! Mình có thể giúp gì cho bạn?', meta: timeNow()});

function sendMessage() {
  const val = input.value.trim();
  if (!val) return;
  addMessage({ who: 'me', text: val, meta: timeNow() });
  input.value = '';
  showTyping();

  const endpoint = ragSwitch.checked
  ? "http://127.0.0.1:8000/ask_rag"
  : "http://127.0.0.1:8000/ask";

  // Gọi mô hình QA backend
  fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: val })
  })
    .then(res => res.json())
    .then(data => {
      hideTyping();
      addMessage({
        who: 'you',
        text: data.answer || 'Không tìm thấy câu trả lời.',
        meta: timeNow()
      });
    })
    .catch(err => {
      hideTyping();
      addMessage({
        who: 'you',
        text: '⚠️ Lỗi kết nối tới server QA',
        meta: timeNow()
      });
      // addMessage({ who: 'you', text: '⚠️ Lỗi kết nối tới server QA.' });
      // backendReadyFlag = false;
      // backendNotReady();
      // checkBackend();
      // backendCheckInterval = setInterval(checkBackend, 5000);
    });
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Hiệu ứng typing
let typingEl = null;
function showTyping() {
  if (typingEl) return;
  typingEl = document.createElement('div');
  typingEl.className = 'row you';
  const av = document.createElement('div');
  av.className = 'avatar-small';
  av.innerHTML = `
        <img
            src="../../images/science_icon.png"
            alt="Avatar"
            style="
            width:100%;
            height:100%;
            border-radius:50%;
            object-fit:cover;
            "
        />
        `;
  typingEl.appendChild(av);

  const wrap = document.createElement('div');
  wrap.className = 'bubble you';
  wrap.innerHTML = '<div class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  typingEl.appendChild(wrap);
  messagesEl.appendChild(typingEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

window.addEventListener('load', () => input.focus());

// async function checkBackend() {
//   if (backendReadyFlag) return; // đã ok thì không cần kiểm tra nữa
//   try {
//     const res = await fetch("http://127.0.0.1:8000/health");
//     const data = await res.json();
//     if (data.status === "ok") {
//       backendReady();
//       backendReadyFlag = true;
//       clearInterval(backendCheckInterval);
//     } else {
//       backendNotReady();
//     }
//   } catch (err) {
//     backendNotReady();
//   }
// }

// Khi backend sẵn sàng
function backendReady() {
  statusDot.style.backgroundColor = "#22c55e"; // xanh lá
  statusText.innerHTML = '<span class="dot" style="background-color:#22c55e"></span> Online';
  composer.style.pointerEvents = "auto";
  input.disabled = false;

  const prepMsg = document.getElementById("prepMsg");
  if (prepMsg) prepMsg.remove();
}

// Khi backend chưa sẵn sàng
function backendNotReady() {
  statusDot.style.backgroundColor = "#facc15"; // vàng
  statusText.innerHTML = '<span class="dot" style="background-color:#facc15"></span> Đang kết nối ....';
  composer.style.pointerEvents = "none";
  input.disabled = true;

  if (!document.getElementById("prepMsg")) {
    const msg = document.createElement("div");
    msg.id = "prepMsg";
    msg.className = "system-msg";
    msg.textContent = "🟡 Chatbot đang chuẩn bị, vui lòng chờ...";
    messagesEl.appendChild(msg);
  }
}

// Kiểm tra ban đầu và định kỳ 5s/lần cho tới khi ok
// checkBackend();
// backendCheckInterval = setInterval(checkBackend, 5000);
