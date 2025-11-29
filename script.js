// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {
  const startDetectingBtn = document.getElementById('startDetectingBtn');
  const detectionInterface = document.getElementById('detection-interface');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const newsInput = document.getElementById('newsInput');
  const resultDiv = document.getElementById('result');

  // Handle "Start Detecting Now" button click
  if (startDetectingBtn && detectionInterface) {
    startDetectingBtn.addEventListener('click', function (e) {
      // if this is an anchor to detect.html, let navigation happen
      if (startDetectingBtn.tagName === 'A' && startDetectingBtn.getAttribute('href')) {
        return; // default behavior
      }
      e.preventDefault();
      detectionInterface.style.display = 'block';
      detectionInterface.scrollIntoView({ behavior: 'smooth' });
      setTimeout(() => { if (newsInput) newsInput.focus(); }, 500);
    });
  }

  // Handle "Analyze Article" button click
  if (analyzeBtn && newsInput) analyzeBtn.addEventListener('click', function () {
    const articleText = newsInput.value.trim();

    if (!articleText) {
      alert('Please enter some text to analyze.');
      return;
    }

    // Show loading state
    showLoading();

    // Simulate API call to your Streamlit backend
    analyzeArticle(articleText);
  });

  function showLoading() {
    resultDiv.className = 'result-container result-loading';
    resultDiv.innerHTML = '<div class="loading"></div>Analyzing article...';
    resultDiv.style.display = 'block';
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
  }

  function showResult(isReal, confidence = null) {
    resultDiv.style.display = 'block';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Article';

    const confidenceValue = confidence || 85;
    const progressWidth = Math.min(confidenceValue, 100);

    if (isReal) {
      resultDiv.className = 'result-container result-success';
      resultDiv.innerHTML = `
        <svg class="result-icon" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
        </svg>
        <div class="result-title" style="color: #065f46;">Appears Authentic</div>
        <div class="result-confidence">Confidence: ${confidenceValue.toFixed(1)}%</div>
        <div class="result-progress">
          <div class="result-progress-bar result-progress-success" style="width: ${progressWidth}%"></div>
        </div>
        <div class="result-notice">
          <div class="result-notice-header">
            <svg class="result-notice-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
            </svg>
            <div class="result-notice-title">Important Notice:</div>
          </div>
          <p class="result-notice-text">This analysis is provided by AI and should not be considered as absolute truth. Always verify information from multiple reliable sources. The accuracy of detection depends on various factors including article length, writing style, and context.</p>
        </div>
      `;
    } else {
      resultDiv.className = 'result-container result-danger';
      resultDiv.innerHTML = `
        <svg class="result-icon" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
        </svg>
        <div class="result-title" style="color: #991b1b;">Potentially Fake News</div>
        <div class="result-confidence">Confidence: ${confidenceValue.toFixed(1)}%</div>
        <div class="result-progress">
          <div class="result-progress-bar result-progress-danger" style="width: ${progressWidth}%"></div>
        </div>
        <div class="result-notice">
          <div class="result-notice-header">
            <svg class="result-notice-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
            </svg>
            <div class="result-notice-title">Important Notice:</div>
          </div>
          <p class="result-notice-text">This analysis is provided by AI and should not be considered as absolute truth. Always verify information from multiple reliable sources. The accuracy of detection depends on various factors including article length, writing style, and context.</p>
        </div>
      `;
    }
  }

  function showError(message) {
    resultDiv.className = 'result-container result-loading';
    resultDiv.innerHTML = `❌ Error: ${message}`;
    resultDiv.style.display = 'block';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Article';
  }

  async function analyzeArticle(text) {
    try {
      // Make API call to the backend server
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      showResult(data.isReal, data.confidence);

    } catch (error) {
      console.error('Error analyzing article:', error);
      showError('Failed to analyze article. Please try again.');
    }
  }

  // Smooth scrolling for navigation links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add some interactive effects
  const featureCards = document.querySelectorAll('.feature-card');
  featureCards.forEach(card => {
    card.addEventListener('mouseenter', function () {
      this.style.transform = 'translateY(-5px) scale(1.02)';
    });

    card.addEventListener('mouseleave', function () {
      this.style.transform = 'translateY(0) scale(1)';
    });
  });

  // Add typing effect to hero title
  const heroTitle = document.querySelector('.hero-title');
  const titleText = heroTitle.textContent;
  heroTitle.textContent = '';

  let i = 0;
  const typeWriter = () => {
    if (i < titleText.length) {
      heroTitle.textContent += titleText.charAt(i);
      i++;
      setTimeout(typeWriter, 50);
    }
  };

  // Start typing effect after a short delay
  setTimeout(typeWriter, 500);
});
