const questions = [
    {
        question: "1. เป้าหมายหลักของปัญหา CartPole ใน Reinforcement Learning คืออะไร?",
        options: [
            "ทำให้รถเข็นวิ่งไปถึงเส้นชัยให้เร็วที่สุด",
            "ทำให้ท่อนไม้ทรงตัวอยู่บนรถเข็นได้นานที่สุด",
            "ทำให้ท่อนไม้ล้มลงให้เร็วที่สุด",
            "เก็บเหรียญรางวัลที่หล่นมาจากฟ้า"
        ],
        answer: 1
    },
    {
        question: "2. ใน CartPole Environment มี Action (การกระทำ) ที่ Agent สามารถเลือกทำได้กี่แบบ?",
        options: [
            "2 แบบ (ดันซ้าย, ดันขวา)",
            "3 แบบ (ดันซ้าย, ดันขวา, หยุดนิ่ง)",
            "4 แบบ (ขึ้น, ลง, ซ้าย, ขวา)",
            "เป็นค่าต่อเนื่อง (Continuous force)"
        ],
        answer: 0
    },
    {
        question: "3. ค่า State (สถานะ) ที่ถูกส่งกลับมาจาก Environment ของ CartPole คืออะไร?",
        options: [
            "ภาพหน้าจอ (Pixels) ของเกม",
            "ข้อความอธิบายสถานการณ์",
            "ค่าพิกัด GPS",
            "ค่าตัวเลข 4 ค่า (เช่น ตำแหน่งรถ, ความเร็วรถ, มุมไม้, ความเร็วมุม)"
        ],
        answer: 3
    },
    {
        question: "4. สมการคณิตศาสตร์ที่เป็นหัวใจสำคัญในการอัปเดตค่า Q-Value เรียกว่าอะไร?",
        options: [
            "สมการนิวตัน (Newton's Equation)",
            "สมการเบลล์แมน (Bellman Equation)",
            "ทฤษฎีบทพีทาโกรัส",
            "สมการของชโรดิงเจอร์"
        ],
        answer: 1
    },
    {
        question: "5. ตัวแปร Gamma (γ) ในสมการ Reinforcement Learning มีหน้าที่อะไร?",
        options: [
            "Discount Factor เพื่อลดทอนความสำคัญของรางวัลในอนาคต",
            "Learning Rate กำหนดความเร็วในการเรียนรู้",
            "ค่าน้ำหนัก (Weight) ของโมเดล",
            "เป็นตัวแปรสุ่มเพื่อเพิ่มการสำรวจ (Exploration)"
        ],
        answer: 0
    },
    {
        question: "6. ทำไมปัญหา CartPole นิยมใช้ Deep Q-Network (DQN) แทนที่จะใช้ตาราง Q-Table ธรรมดา?",
        options: [
            "เพราะ Q-Table เขียนโปรแกรมยากกว่ามาก",
            "เพราะมีจำนวน Action มากเป็นอนันต์",
            "เพราะ State เป็นค่าต่อเนื่อง (Continuous) ทำให้ตารางมีขนาดใหญ่เกินไป",
            "เพราะ DQN ทำงานได้เร็วกว่า Q-Table 100 เท่า"
        ],
        answer: 2
    },
    {
        question: "7. กลยุทธ์ Epsilon-Greedy (ε-greedy) ใช้เพื่อวัตถุประสงค์ใดในการฝึก?",
        options: [
            "เพื่อให้ Agent แข่งขันกันเอง",
            "เพื่อรักษาสมดุลระหว่างการสำรวจสิ่งใหม่ (Exploration) และการใช้ความรู้เดิม (Exploitation)",
            "เพื่อเพิ่มจำนวน Reward สูงสุดที่จะได้รับ",
            "เพื่อลดการใช้หน่วยความจำ"
        ],
        answer: 1
    },
    {
        question: "8. โครงสร้างของ Neural Network พื้นฐานที่ใช้กับ CartPole (DQN) มักจะเป็นรูปแบบใด?",
        options: [
            "Convolutional Neural Network (CNN)",
            "Recurrent Neural Network (RNN)",
            "Dense Layer (Fully Connected Layer) เพียง 2-3 ชั้น",
            "Transformer Network"
        ],
        answer: 2
    },
    {
        question: "9. Reward ของเกม CartPole จะเป็นอย่างไรในแต่ละ Step ที่ท่อนไม้ยังไม่ล้ม?",
        options: [
            "ได้รับ +1 ในทุกๆ Step ที่รอดชีวิต",
            "ได้รับ -1 เพื่อกระตุ้นให้รีบจบเกม",
            "ได้รับคะแนนตามระยะทางที่รถเข็นวิ่งได้",
            "ไม่มี Reward จนกว่าเกมจะโอเวอร์"
        ],
        answer: 0
    },
    {
        question: "10. หากไม้เอียงเกินองศาที่กำหนด (เช่น เอียงเกิน 12 องศา) จะเกิดอะไรขึ้น?",
        options: [
            "Agent จะได้รับโบนัสพิเศษ",
            "ถือว่ารอบนั้นสิ้นสุดลง (Done = True) ต้องเริ่มรอบใหม่",
            "สภาพแวดล้อมจะหยุดชั่วคราวให้คิด",
            "รถเข็นจะถูกผลักกลับมาตรงกลางอัตโนมัติ"
        ],
        answer: 1
    }
];

let currentQuestion = 0;
let score = 0;

const quizBody = document.getElementById('quiz-body');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const resultModal = document.getElementById('result-modal');
const resultTitle = document.getElementById('result-title');
const resultScore = document.getElementById('result-score');
const restartBtn = document.getElementById('restart-btn');

function loadQuestion() {
    const q = questions[currentQuestion];
    quizBody.innerHTML = `
        <div class="question-text">${q.question}</div>
        <div class="options">
            ${q.options.map((opt, index) => `
                <button class="option-btn" onclick="checkAnswer(${index}, this)">
                    ${String.fromCharCode(65 + index)}. ${opt}
                </button>
            `).join('')}
        </div>
    `;
    
    const progress = ((currentQuestion) / questions.length) * 100;
    progressBar.style.width = `${progress}%`;
    progressText.innerText = `ข้อ ${currentQuestion + 1} / ${questions.length}`;
}

window.checkAnswer = function(selectedIndex, btnElement) {
    const q = questions[currentQuestion];
    const allBtns = document.querySelectorAll('.option-btn');
    
    // Disable all buttons
    allBtns.forEach(btn => btn.disabled = true);
    
    if (selectedIndex === q.answer) {
        btnElement.classList.add('correct');
        score++;
    } else {
        btnElement.classList.add('wrong');
        allBtns[q.answer].classList.add('correct'); // Highlight correct answer
    }
    
    setTimeout(() => {
        currentQuestion++;
        if (currentQuestion < questions.length) {
            loadQuestion();
        } else {
            showResult();
        }
    }, 1500); // Wait 1.5s before next question
}

function showResult() {
    progressBar.style.width = `100%`;
    progressText.innerText = `เสร็จสิ้น`;
    
    resultScore.innerText = `คุณทำได้ ${score} / ${questions.length} คะแนน`;
    
    if (score === 10) {
        resultTitle.innerText = "🏆 สมบูรณ์แบบ!";
    } else if (score >= 7) {
        resultTitle.innerText = "🌟 ยอดเยี่ยม!";
    } else if (score >= 5) {
        resultTitle.innerText = "👍 ผ่านเกณฑ์";
    } else {
        resultTitle.innerText = "📚 ต้องพยายามอีกนิด";
    }
    
    resultModal.classList.remove('hidden');
}

restartBtn.addEventListener('click', () => {
    currentQuestion = 0;
    score = 0;
    resultModal.classList.add('hidden');
    loadQuestion();
});

// Initialize
loadQuestion();
