/**
 * Navigation Interceptor
 * Intercepts all navigation links and buttons to show loading page
 */

(function() {
    'use strict';

    // Function to redirect through loading page
    function redirectWithLoading(targetUrl) {
        // Don't intercept if already on loading page
        if (window.location.pathname.includes('loading.html')) {
            return false;
        }

        // Don't intercept anchor links (same page navigation)
        if (targetUrl.startsWith('#')) {
            return false;
        }

        // Don't intercept external links
        if (targetUrl.startsWith('http://') || 
            targetUrl.startsWith('https://') || 
            targetUrl.startsWith('mailto:') || 
            targetUrl.startsWith('tel:')) {
            return false;
        }

        // Don't intercept if already going to loading page
        if (targetUrl.includes('loading.html')) {
            return false;
        }

        // Redirect through loading page
        window.location.href = `loading.html?target=${encodeURIComponent(targetUrl)}`;
        return true;
    }

    // Initialize when DOM is ready
    function init() {
        // Intercept all anchor tag clicks
        document.addEventListener('click', function(e) {
            const anchor = e.target.closest('a');
            
            if (anchor && anchor.href) {
                const href = anchor.getAttribute('href');
                
                // Check if it's a local navigation link
                if (href && 
                    !href.startsWith('#') && 
                    !href.startsWith('http://') && 
                    !href.startsWith('https://') && 
                    !href.startsWith('mailto:') && 
                    !href.startsWith('tel:') &&
                    !href.includes('loading.html')) {
                    
                    // Check if it's a relative path (local file)
                    try {
                        const url = new URL(href, window.location.origin);
                        const currentUrl = new URL(window.location.href);
                        
                        // If same origin, intercept it
                        if (url.origin === currentUrl.origin || !href.includes('://')) {
                            e.preventDefault();
                            e.stopPropagation();
                            redirectWithLoading(href);
                            return false;
                        }
                    } catch (e) {
                        // If URL parsing fails, assume it's a relative path
                        e.preventDefault();
                        e.stopPropagation();
                        redirectWithLoading(href);
                        return false;
                    }
                }
            }
        }, true);
    }

    // Helper function for programmatic navigation
    window.navigateWithLoading = function(url) {
        redirectWithLoading(url);
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
