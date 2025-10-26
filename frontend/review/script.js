const messagesEl = document.getElementById('messages');
const reviewInput = document.getElementById('reviewInput');
const aspectInput = document.getElementById('aspectInput');
const evalBtn = document.getElementById('evalBtn');
const statusDot = document.querySelector('.status .dot');
const statusText = document.querySelector('.status');
const composer = document.getElementById('composer');

// let backendReady = false;
// let checkInterval = null;

// 🟢 Kiểm tra backend
// async function checkBackend() {
//   try {
//     const res = await fetch("http://127.0.0.1:8000/health");
//     const data = await res.json();

//     if (data.status === "ok") {
//       backendReady = true;
//       statusDot.style.backgroundColor = "#22c55e"; // xanh lá
//       statusText.innerHTML = '<span class="dot" style="background-color:#22c55e"></span> Online';
//       composer.style.pointerEvents = "auto";
//       reviewInput.disabled = false;
//       aspectInput.disabled = false;
//       evalBtn.disabled = false;

//       const waitingMsg = document.getElementById("waitingMsg");
//       if (waitingMsg) waitingMsg.remove();
//       clearInterval(checkInterval);
//     } else {
//       backendNotReady();
//     }
//   } catch (err) {
//     backendNotReady();
//   }
// }

// 🟡 Khi backend chưa sẵn sàng hoặc lỗi
// function backendNotReady() {
//   backendReady = false;
//   statusDot.style.backgroundColor = "#facc15"; // vàng
//   statusText.innerHTML = '<span class="dot" style="background-color:#facc15"></span> Đang kết nối...';
//   composer.style.pointerEvents = "none";
//   reviewInput.disabled = true;
//   aspectInput.disabled = true;
//   evalBtn.disabled = true;

//   if (!document.getElementById("waitingMsg")) {
//     const msg = document.createElement("div");
//     msg.id = "waitingMsg";
//     msg.className = "system-msg";
//     msg.textContent = "🟡 Đang kết nối tới backend, vui lòng chờ...";
//     messagesEl.appendChild(msg);
//   }

//   if (!checkInterval) checkInterval = setInterval(checkBackend, 5000);
// }

// 🧠 Thêm khung kết quả cảm xúc
evalBtn.disabled = false
function addMessage(text, sentiment) {
  const msg = document.createElement('div');
  msg.className = 'sentiment-msg';
  msg.textContent = text;

  if (sentiment === 'positive') msg.style.backgroundColor = '#22c55e33';
  else if (sentiment === 'negative') msg.style.backgroundColor = '#ef444433';
  else msg.style.backgroundColor = '#9ca3af33';

  msg.style.borderLeft = `6px solid ${
    sentiment === 'positive' ? '#22c55e' :
    sentiment === 'negative' ? '#ef4444' : '#9ca3af'
  }`;

  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// 🚀 Gọi API phân tích cảm xúc
async function predictSentiment(review, aspect) {
  try {
    const res = await fetch("http://127.0.0.1:8000/predict_sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: review, aspect: aspect })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.sentiment || 'neutral';
  } catch (err) {
    // backendReady = false;
    // backendNotReady();
    // checkInterval = setInterval(checkBackend, 5000);
    return 'error'
  }
}

// 📤 Khi người dùng nhấn Đánh giá
evalBtn.addEventListener('click', async () => {
  const review = reviewInput.value.trim();
  const aspect = aspectInput.value.trim();
  if (!review || !aspect ) return;

  const loading = document.createElement('div');
  loading.className = 'system-msg';
  loading.textContent = '⏳ Đang đánh giá cảm xúc...';
  messagesEl.appendChild(loading);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const sentiment = await predictSentiment(review, aspect);
  loading.remove();

  const text = `Bình luận: ${review}\n\nKhía cạnh: ${aspect}`;
  if (sentiment == 'error')
    text = "Lỗi kết nối với server"
  addMessage(text, sentiment);

  reviewInput.value = '';
  aspectInput.value = '';
});

// 🟡 Kiểm tra backend khi khởi động
// checkBackend();
// checkInterval = setInterval(checkBackend, 5000);
