# TODO: Add Clear History Functionality

## Tasks
- [x] Add new API endpoint `/api/clear-history` in server.py to delete all analysis_history for authenticated user
- [x] Update clearActivityBtn event listener in profile.html to call server API if logged in, clear localStorage, reload timeline, hide loadMoreActivityBtn, and show success message
- [x] Test the functionality to ensure both local and server-side history is cleared and load more button is removed (server running successfully, browser testing disabled)
