document.addEventListener('DOMContentLoaded', function() {
    const taskInput = document.getElementById('taskInput');
    const submitBtn = document.getElementById('submitBtn');
    const chatBox = document.getElementById('chatBox');

    submitBtn.addEventListener('click', async function() {
        const task = taskInput.value.trim();
        if (!task) return;

        // Append user message
        appendMessage('User', task);

        try {
            const response = await fetch('/orchestrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: task })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            // Escape output for XSS
            const escapedSteps = data.steps.map(step => step.replace(/</g, '&lt;').replace(/>/g, '&gt;')).join('<br>');
            const escapedResponse = data.response.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const output = `Steps: ${escapedSteps}<br><strong>Response:</strong><br>${escapedResponse}`;
            appendMessage('Office-Cube', output);
        } catch (error) {
            appendMessage('Error', `Failed: ${error.message}`);
        }

        taskInput.value = '';
    });

    function appendMessage(sender, message) {
        const div = document.createElement('div');
        div.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Focus styles for accessibility
    submitBtn.addEventListener('focus', () => submitBtn.style.outline = '2px solid #007bff');
});
