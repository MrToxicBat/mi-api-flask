const revealElements = document.querySelectorAll('.reveal');
const heroGlow = document.querySelector('.hero-glow');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 }
);

revealElements.forEach((element) => observer.observe(element));

window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  if (heroGlow) {
    heroGlow.style.transform = `translateY(${scrollY * 0.25}px)`;
  }
});
