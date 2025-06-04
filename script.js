// Initialize Swiper for the hero slider
// Make sure to uncomment the Swiper CSS and JS links in index.html

var heroSwiper = new Swiper('.hero-slider .swiper-container', {
    // Optional parameters
    direction: 'horizontal',
    loop: true,
    autoplay: {
        delay: 5000, // Change slides every 5 seconds
        disableOnInteraction: false,
    },
    effect: 'fade', // Optional: add a fade effect
    fadeEffect: {
        crossFade: true
    },

    // If we need pagination
    pagination: {
        el: '.hero-slider .swiper-pagination',
        clickable: true,
    },

    // Navigation arrows
    navigation: {
        nextEl: '.hero-slider .swiper-button-next',
        prevEl: '.hero-slider .swiper-button-prev',
    },
});


// You can add more JavaScript here for other interactions or animations 