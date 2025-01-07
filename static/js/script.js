$(document).ready(function() {
    $('#send-btn').click(function() {
        var userMessage = $('#user-input').val().trim();
        if (userMessage === "") return;

        // Display user message
        $('#chat-box').append('<div class="message user-msg">' + userMessage + '</div>');
        $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);

        // Clear input field
        $('#user-input').val('');

        // Send message to backend
        $.ajax({
            url: '/generate',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ "prompt": userMessage }),
            success: function(response) {
                var botResponse = response.response;
                $('#chat-box').append('<div class="message bot-msg">' + botResponse + '</div>');
                $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
            },
            error: function(xhr, status, error) {
                $('#chat-box').append('<div class="message bot-msg">Error: ' + error + '</div>');
                $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
            }
        });
    });

    // Allow 'Enter' key to send messages
    $('#user-input').keypress(function(e) {
        if (e.which === 13) {
            $('#send-btn').click();
        }
    });
});
