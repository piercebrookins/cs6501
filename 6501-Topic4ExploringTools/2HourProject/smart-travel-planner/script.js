/**
 * Nexus Website - Interactive Elements
 * Based on Ultimate Website Design Guide v5
 */

// ==========================================================================
// CONFIGURATION
// ==========================================================================

const API_BASE_URL = '/api';  // Relative URL since served from same origin

// ==========================================================================
// MOBILE NAVIGATION
// ==========================================================================

const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const mobileNav = document.querySelector('.mobile-nav');
const mobileNavLinks = document.querySelectorAll('.mobile-nav a');

if (mobileMenuBtn && mobileNav) {
  mobileMenuBtn.addEventListener('click', () => {
    mobileNav.classList.toggle('open');
    mobileMenuBtn.classList.toggle('active');
    document.body.style.overflow = mobileNav.classList.contains('open') ? 'hidden' : '';
  });

  // Close mobile nav when clicking a link
  mobileNavLinks.forEach(link => {
    link.addEventListener('click', () => {
      mobileNav.classList.remove('open');
      mobileMenuBtn.classList.remove('active');
      document.body.style.overflow = '';
    });
  });

  // Close on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mobileNav.classList.contains('open')) {
      mobileNav.classList.remove('open');
      mobileMenuBtn.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
}

// ==========================================================================
// SCROLL ANIMATIONS (Intersection Observer)
// ==========================================================================

const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.1
};

const animationObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      // Optionally unobserve after animation to save resources
      // animationObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe all elements with animation classes
const animatedElements = document.querySelectorAll('.fade-in, .stagger-children');
animatedElements.forEach(el => animationObserver.observe(el));

// ==========================================================================
// HEADER SCROLL EFFECT
// ==========================================================================

const header = document.querySelector('.header');
let lastScrollY = window.scrollY;
let ticking = false;

function updateHeader() {
  const currentScrollY = window.scrollY;
  
  // Add shadow on scroll
  if (currentScrollY > 10) {
    header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
  } else {
    header.style.boxShadow = 'none';
  }
  
  // Hide/show header on scroll direction (optional - uncomment to enable)
  /*
  if (currentScrollY > lastScrollY && currentScrollY > 100) {
    header.style.transform = 'translateY(-100%)';
  } else {
    header.style.transform = 'translateY(0)';
  }
  */
  
  lastScrollY = currentScrollY;
  ticking = false;
}

window.addEventListener('scroll', () => {
  if (!ticking) {
    requestAnimationFrame(updateHeader);
    ticking = true;
  }
});

// ==========================================================================
// SMOOTH SCROLL FOR ANCHOR LINKS
// ==========================================================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    
    // Skip if it's just "#"
    if (href === '#') return;
    
    const target = document.querySelector(href);
    if (target) {
      e.preventDefault();
      
      const headerHeight = header ? header.offsetHeight : 0;
      const targetPosition = target.getBoundingClientRect().top + window.scrollY - headerHeight - 20;
      
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  });
});

// ==========================================================================
// BUTTON RIPPLE EFFECT (Optional enhancement)
// ==========================================================================

function createRipple(event) {
  const button = event.currentTarget;
  
  // Remove existing ripple
  const existingRipple = button.querySelector('.ripple');
  if (existingRipple) {
    existingRipple.remove();
  }
  
  const ripple = document.createElement('span');
  ripple.classList.add('ripple');
  
  const rect = button.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  
  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
  ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
  
  button.appendChild(ripple);
  
  // Remove ripple after animation
  ripple.addEventListener('animationend', () => {
    ripple.remove();
  });
}

// Add ripple to primary buttons
document.querySelectorAll('.btn-primary').forEach(button => {
  button.style.position = 'relative';
  button.style.overflow = 'hidden';
  button.addEventListener('click', createRipple);
});

// Add ripple styles dynamically
const rippleStyles = document.createElement('style');
rippleStyles.textContent = `
  .ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple-animation 0.6s linear;
    pointer-events: none;
  }
  
  @keyframes ripple-animation {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`;
document.head.appendChild(rippleStyles);

// ==========================================================================
// PARALLAX FLOATING CARDS (Subtle mouse movement)
// ==========================================================================

const floatingCards = document.querySelectorAll('.floating-card');

if (floatingCards.length > 0) {
  document.addEventListener('mousemove', (e) => {
    const mouseX = e.clientX / window.innerWidth - 0.5;
    const mouseY = e.clientY / window.innerHeight - 0.5;
    
    floatingCards.forEach((card, index) => {
      const speed = (index + 1) * 10;
      const x = mouseX * speed;
      const y = mouseY * speed;
      
      card.style.transform = `translate(${x}px, ${y}px)`;
    });
  });
}

// ==========================================================================
// PRICING CARD HOVER EFFECT
// ==========================================================================

const pricingCards = document.querySelectorAll('.pricing-card');

pricingCards.forEach(card => {
  card.addEventListener('mouseenter', () => {
    // Dim other cards slightly
    pricingCards.forEach(otherCard => {
      if (otherCard !== card) {
        otherCard.style.opacity = '0.7';
      }
    });
  });
  
  card.addEventListener('mouseleave', () => {
    // Reset all cards
    pricingCards.forEach(otherCard => {
      otherCard.style.opacity = '1';
    });
  });
});

// ==========================================================================
// TYPED EFFECT FOR HERO (Optional - uncomment to enable)
// ==========================================================================

/*
const heroTitle = document.querySelector('.hero-title');
const originalText = heroTitle.innerHTML;

function typeText(element, text, speed = 50) {
  element.innerHTML = '';
  let i = 0;
  
  function type() {
    if (i < text.length) {
      element.innerHTML += text.charAt(i);
      i++;
      setTimeout(type, speed);
    }
  }
  
  type();
}

// Trigger on page load
window.addEventListener('load', () => {
  typeText(heroTitle, originalText, 30);
});
*/

// ==========================================================================
// COUNTER ANIMATION FOR STATS
// ==========================================================================

function animateCounter(element, target, duration = 2000) {
  let start = 0;
  const increment = target / (duration / 16);
  
  function updateCounter() {
    start += increment;
    if (start < target) {
      element.textContent = Math.floor(start).toLocaleString();
      requestAnimationFrame(updateCounter);
    } else {
      element.textContent = target.toLocaleString();
    }
  }
  
  updateCounter();
}

// Observe stats for animation trigger
const statValues = document.querySelectorAll('.stat-value');
const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const text = entry.target.textContent;
      
      // Only animate numeric values
      if (text.includes('+')) {
        const num = parseInt(text.replace(/[^0-9]/g, ''));
        entry.target.textContent = '0+';
        animateCounter(entry.target, num);
        entry.target.textContent = num.toLocaleString() + '+';
      }
      
      statsObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

statValues.forEach(stat => statsObserver.observe(stat));

// ==========================================================================
// FORM INTERACTIONS (If forms are added later)
// ==========================================================================

function initForms() {
  const inputs = document.querySelectorAll('input, textarea');
  
  inputs.forEach(input => {
    // Float label effect
    input.addEventListener('focus', () => {
      input.parentElement?.classList.add('focused');
    });
    
    input.addEventListener('blur', () => {
      if (!input.value) {
        input.parentElement?.classList.remove('focused');
      }
    });
    
    // Validate on blur
    input.addEventListener('blur', () => {
      if (input.hasAttribute('required') && !input.value) {
        input.classList.add('error');
      } else {
        input.classList.remove('error');
      }
    });
  });
}

// Initialize forms if they exist
if (document.querySelector('form')) {
  initForms();
}

// ==========================================================================
// CONSOLE EASTER EGG
// ==========================================================================

console.log(`
%c🌤️ TRIPCAST %c

Welcome, curious developer! 👋
This site was built following the Ultimate Website Design Guide v5.

Key principles applied:
• Dark sophisticated theme with teal accents
• 8px spacing grid system  
• Two-font typography (DM Sans + Libre Baskerville)
• Scroll-triggered animations
• Mobile-first responsive design

Powered by OpenWeatherMap API.
Built with ❤️ for the Smart Travel Planner project.
`, 
'color: #00D4AA; font-size: 24px; font-weight: bold;',
'color: #888; font-size: 12px;'
);

// ==========================================================================
// TRY IT OUT - TRIP PLANNER FORM
// ==========================================================================

const tripForm = document.getElementById('trip-form');
const destinationInput = document.getElementById('destination');
const unitsSelect = document.getElementById('units');
const resultsContainer = document.getElementById('results');
const weatherResults = document.getElementById('weather-results');
const packingResults = document.getElementById('packing-results');
const activityResults = document.getElementById('activity-results');
const resultsTitle = document.getElementById('results-title');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');
const clearResultsBtn = document.getElementById('clear-results');
const retryBtn = document.getElementById('retry-btn');
const cityButtons = document.querySelectorAll('.city-btn');

// Form submission handler
if (tripForm) {
  tripForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const destination = destinationInput.value.trim();
    const units = unitsSelect.value;
    
    if (!destination) {
      showError('Please enter a destination city.');
      return;
    }
    
    await fetchTripPlan(destination, units);
  });
}

// Quick city buttons
cityButtons.forEach(btn => {
  btn.addEventListener('click', async () => {
    const city = btn.dataset.city;
    destinationInput.value = city;
    await fetchTripPlan(city, unitsSelect.value);
  });
});

// Clear results
if (clearResultsBtn) {
  clearResultsBtn.addEventListener('click', () => {
    hideResults();
    destinationInput.value = '';
    destinationInput.focus();
  });
}

// Retry button
if (retryBtn) {
  retryBtn.addEventListener('click', () => {
    hideError();
    destinationInput.focus();
  });
}

// Fetch trip plan from API (uses OpenAI when available, falls back to hardcoded)
async function fetchTripPlan(city, units = 'metric') {
  showLoading(true);
  hideError();
  hideResults();
  
  try {
    const response = await fetch(`${API_BASE_URL}/travel-plan?city=${encodeURIComponent(city)}&units=${units}`);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch weather data');
    }
    
    displayResults(city, data);
    
  } catch (error) {
    console.error('Error fetching trip plan:', error);
    showError(error.message || 'Unable to fetch weather data. Please try again.');
  } finally {
    showLoading(false);
  }
}

// Display results — uses AI data when available, hardcoded fallback otherwise
function displayResults(city, data) {
  resultsTitle.textContent = `🌤️ Trip Plan for ${city}`;
  
  // Display raw weather data
  weatherResults.textContent = data.weather || 'No weather data available';

  // Show an AI badge or fallback note
  const badge = data.ai_powered
    ? '<span class="ai-badge">✨ AI-Powered</span>'
    : '<span class="ai-badge fallback">📋 Basic</span>';

  if (data.ai_powered && data.summary) {
    // Inject a summary block above the cards
    const existingSummary = document.getElementById('ai-summary');
    if (existingSummary) existingSummary.remove();

    const summaryEl = document.createElement('div');
    summaryEl.id = 'ai-summary';
    summaryEl.className = 'ai-summary';
    summaryEl.innerHTML = `<p>${badge} ${data.summary}</p>`;
    weatherResults.after(summaryEl);
  } else {
    const existingSummary = document.getElementById('ai-summary');
    if (existingSummary) existingSummary.remove();
  }

  // Packing list — prefer AI, fall back to hardcoded
  const packingItems = (data.ai_powered && data.packing)
    ? data.packing
    : generatePackingListItems(data);
  packingResults.querySelector('.results-content').innerHTML =
    `<ul>${packingItems.map(item => `<li>${item}</li>`).join('')}</ul>`;

  // Activities — prefer AI, fall back to hardcoded
  const activityItems = (data.ai_powered && data.activities)
    ? data.activities
    : generateActivityItems(data);
  activityResults.querySelector('.results-content').innerHTML =
    `<ul>${activityItems.map(a => `<li>${a}</li>`).join('')}</ul>`;
  
  resultsContainer.hidden = false;
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------------------
// FALLBACK generators — used when OpenAI is unavailable
// ---------------------------------------------------------------------------

// Returns a flat array of packing item strings
function generatePackingListItems(data) {
  const items = [];
  const weather = data.weather || '';
  
  const tempMatch = weather.match(/(-?\d+\.?\d*).*?(-?\d+\.?\d*).*?°/);
  let avgTemp = 20;
  if (tempMatch) {
    avgTemp = (parseFloat(tempMatch[1]) + parseFloat(tempMatch[2])) / 2;
  }
  
  const hasRain = weather.toLowerCase().includes('rain') || 
                  weather.match(/\d{2,3}%.*rain/i);
  
  if (avgTemp < 10) {
    items.push('Heavy winter coat', 'Thermal layers', 'Warm sweater', 'Gloves & scarf', 'Warm boots', 'Beanie/warm hat');
  } else if (avgTemp < 20) {
    items.push('Light jacket or fleece', 'Long pants/jeans', 'Layerable tops', 'Closed-toe shoes', 'Light sweater');
  } else if (avgTemp < 30) {
    items.push('Light clothing', 'Shorts & t-shirts', 'Comfortable sandals', 'Light cardigan', 'Walking shoes');
  } else {
    items.push('Very light fabrics', 'Shorts & tank tops', 'Wide-brim sun hat', 'Light colored clothing', 'Breathable shoes');
  }
  
  if (hasRain) {
    items.push('Compact umbrella', 'Rain jacket', 'Waterproof shoes');
  }
  items.push('Sunscreen (SPF 30+)', 'Sunglasses');
  return items;
}

// Returns a flat array of activity strings
function generateActivityItems(data) {
  const activities = [];
  const weather = data.weather || '';
  
  const hasRain = weather.toLowerCase().includes('rain') || 
                  weather.match(/\d{2,3}%.*rain/i);
  const isSunny = weather.toLowerCase().includes('clear') || 
                  weather.toLowerCase().includes('sunny');
  
  const tempMatch = weather.match(/(-?\d+\.?\d*).*?(-?\d+\.?\d*).*?°/);
  let avgTemp = 20;
  if (tempMatch) {
    avgTemp = (parseFloat(tempMatch[1]) + parseFloat(tempMatch[2])) / 2;
  }
  
  if (hasRain) {
    activities.push('Visit museums & galleries', 'Explore local cafés', 'Indoor shopping', 'Cooking class', 'Spa day', 'Catch a local show');
  }
  if (isSunny && avgTemp > 15) {
    activities.push('Walking city tour', 'Park picnic', 'Outdoor markets', 'Rooftop bars', 'Bike rental', 'Scenic viewpoints');
  }
  if (avgTemp < 10) {
    activities.push('Hot springs or thermal baths', 'Cozy restaurant hopping', 'Indoor cultural sites', 'Local coffee shops');
  }
  if (avgTemp > 28) {
    activities.push('Swimming/beach time', 'Early morning sightseeing', 'Air-conditioned museums', 'Evening outdoor dining');
  }
  if (activities.length === 0) {
    activities.push('Explore the city center', 'Visit local landmarks', 'Try local cuisine', 'People-watch at a café', 'Take photos at scenic spots');
  }
  return activities;
}

// Show/hide loading state
function showLoading(isLoading) {
  const submitBtn = tripForm?.querySelector('button[type="submit"]');
  if (!submitBtn) return;
  
  const btnText = submitBtn.querySelector('.btn-text');
  const btnLoading = submitBtn.querySelector('.btn-loading');
  const btnArrow = submitBtn.querySelector('.btn-arrow');
  
  if (isLoading) {
    btnText.hidden = true;
    btnLoading.hidden = false;
    btnArrow.hidden = true;
    submitBtn.disabled = true;
  } else {
    btnText.hidden = false;
    btnLoading.hidden = true;
    btnArrow.hidden = false;
    submitBtn.disabled = false;
  }
}

// Show error
function showError(message) {
  if (errorContainer && errorMessage) {
    errorMessage.textContent = message;
    errorContainer.hidden = false;
    resultsContainer.hidden = true;
  }
}

// Hide error
function hideError() {
  if (errorContainer) {
    errorContainer.hidden = true;
  }
}

// Hide results
function hideResults() {
  if (resultsContainer) {
    resultsContainer.hidden = true;
  }
}

// ==========================================================================
// PERFORMANCE: Defer non-critical operations
// ==========================================================================

window.addEventListener('load', () => {
  // Preload fonts for better performance
  const fonts = [
    'DM Sans',
    'Libre Baskerville'
  ];
  
  fonts.forEach(font => {
    document.fonts.ready.then(() => {
      console.log(`Font loaded: ${font}`);
    });
  });
});