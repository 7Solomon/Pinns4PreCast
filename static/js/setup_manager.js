const SetupManager = {
    editors: {},

    init: function () {
        document.addEventListener('DOMContentLoaded', () => {
            // Safety check for data
            if (!window.SCHEMAS || !window.FILES) {
                console.error("Critical: SCHEMAS or FILES not loaded from server.");
                return;
            }
            this.instantiateEditors();
        });
    },

    instantiateEditors: function () {
        // Find all divs designated for editors
        const containers = document.querySelectorAll('.editor-mount-point');

        containers.forEach(container => {
            const type = container.dataset.type;
            const filename = container.dataset.filename;
            const containerId = container.id;

            // 1. Get the specific schema
            const schema = window.SCHEMAS[type];

            // 2. Get the specific file content from the global object
            const initialValue = window.FILES[type][filename];

            if (!schema) {
                console.error(`Missing schema for type: ${type}`);
                return;
            }

            // 3. Initialize JSON Editor
            const editor = new JSONEditor(container, {
                schema: schema,
                startval: initialValue,
                theme: 'bootstrap5',
                iconlib: 'bootstrap-icons',

                disable_edit_json: true,
                disable_properties: true,

                disable_collapse: true,

                object_layout: 'normal',

                // Clean up array controls 
                disable_array_add: true,
                disable_array_delete: true,
                disable_array_reorder: true,

                form_name_root: 'Parameters'
            });

            this.editors[containerId] = editor;
        });
    },

    save: function (type, filename, containerId) {
        const editor = this.editors[containerId];
        if (!editor) {
            alert("Editor instance not found. Try refreshing the page.");
            return;
        }

        // Validate
        const errors = editor.validate();
        if (errors.length) {
            alert(`Validation errors found (${errors.length}). Please check the red fields.`);
            return;
        }

        const content = editor.getValue();

        fetch('/info/save_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, filename, content })
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert("Server Error: " + data.error);
                } else {
                    this.showToast(`${filename} saved successfully!`, "success");
                }
            })
            .catch(err => alert("Network Error: " + err));
    },

    setActive: function (type, filename) {
        const payload = { [type]: filename };

        fetch('/info/load_state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                this.showToast("Active state updated!", "success");
                // Reload page to update badges and UI state
                setTimeout(() => location.reload(), 800);
            })
            .catch(err => alert("Error setting active state: " + err));
    },

    createNew: function (type) {
        const name = prompt(`Enter new ${type} filename (without .json):`);
        if (!name) return;

        // Fetch default values first
        fetch(`/info/defaults/${type}`)
            .then(r => r.json())
            .then(defaults => {
                // Then save as new file
                return fetch('/info/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type, filename: name, content: defaults })
                });
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                this.showToast("File created!", "success");
                setTimeout(() => location.reload(), 1000);
            })
            .catch(err => alert("Error creating file: " + err));
    },

    deleteFile: function (type, filename) {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

        this.showToast("Delete functionality requires backend implementation.", "info");
        // fetch(`/info/delete/${type}/${filename}`, { method: 'DELETE' }) ...
    },

    showToast: function (message, type) {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} shadow-lg border-0 fade show`;
        toast.style.cssText = 'position:fixed; top:20px; right:20px; z-index:10000; min-width:300px;';

        let icon = type === 'success' ? '<i class="bi bi-check-circle-fill me-2"></i>' : '<i class="bi bi-info-circle-fill me-2"></i>';

        toast.innerHTML = `
            <div class="d-flex align-items-center justify-content-between">
                <div>${icon} <strong>${message}</strong></div>
                <button type="button" class="btn-close" onclick="this.closest('.alert').remove()"></button>
            </div>
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
};

SetupManager.init();
