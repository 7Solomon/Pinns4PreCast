const SetupManager = {
    jsonEditors: {},

    init: function () {
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeJsonEditors();
        });
    },

    initializeJsonEditors: function () {
        const containers = document.querySelectorAll('.json-editor-container');

        containers.forEach(container => {
            const fieldName = container.dataset.fieldName;
            // Find the hidden input that stores the actual stringified JSON
            const hiddenInput = container.parentElement.querySelector(`input[name="${fieldName}"]`);

            if (!hiddenInput) return;

            let initialValue = {};
            try {
                initialValue = JSON.parse(hiddenInput.value);
            } catch (e) {
                console.error(`Invalid JSON for ${fieldName}`, e);
            }

            const options = {
                mode: 'tree',
                modes: ['tree', 'code'],
                search: true,
                mainMenuBar: true,
                navigationBar: false,
                statusBar: false,
                onChangeText: function (jsonString) {
                    hiddenInput.value = jsonString;
                }
            };

            const editor = new JSONEditor(container, options);
            editor.set(initialValue);
            this.jsonEditors[fieldName] = editor;
        });
    },

    /**
     * Recursively fixes array-like objects (keys "0", "1"...) into real Arrays.
     */
    fixArrays: function (obj) {
        if (typeof obj !== 'object' || obj === null) return obj;

        // Check if object keys look like "0", "1", "2"...
        const keys = Object.keys(obj).sort((a, b) => a - b);
        const isArrayLike = keys.length > 0 && keys.every((k, i) => String(k) === String(i));

        if (isArrayLike) {
            const newArr = [];
            keys.forEach(k => {
                newArr.push(this.fixArrays(obj[k]));
            });
            return newArr;
        }

        for (let k in obj) {
            obj[k] = this.fixArrays(obj[k]);
        }
        return obj;
    },

    saveInline: function (event, type, filename) {
        event.preventDefault();
        const formData = new FormData(event.target);
        let content = {};

        for (let [key, value] of formData.entries()) {
            // 1. Handle JSON Editor Fields
            if (this.jsonEditors[key]) {
                try {
                    const jsonValue = this.jsonEditors[key].get();
                    this.setNestedValue(content, key, jsonValue);
                } catch (e) {
                    alert(`Error in JSON field '${key}':\n${e.message}`);
                    return;
                }
            }
            // 2. Handle Min/Max Range Fields
            else if (key.endsWith('_min')) {
                const baseKey = key.replace('_min', '');
                const max = formData.get(baseKey + '_max');
                this.setNestedValue(content, baseKey, [parseFloat(value), parseFloat(max)]);
            }
            // 3. Handle Standard Fields
            else if (!key.endsWith('_max')) {
                const inputElement = event.target.querySelector(`[name="${key}"]`);
                const isNumber = inputElement && inputElement.type === 'number';
                let parsedValue = value;

                if (isNumber && value !== "") {
                    parsedValue = parseFloat(value);
                }
                // Convert "on" checkbox to boolean true
                if (inputElement && inputElement.type === 'checkbox') {
                    // Note: FormData only includes checked boxes. If handled here, ensure logic covers unchecked.
                    // Usually cleaner to rely on hidden inputs or specific parsing for checkboxes if strictly needed.
                }

                this.setNestedValue(content, key, parsedValue);
            }
            else {
                // Fallback
                this.setNestedValue(content, key, value);
            }
        }

        content = this.fixArrays(content);

        fetch('/info/save_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, filename, content })
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) alert("Error: " + data.error);
                else this.showToast("Changes saved successfully!", "success");
            })
            .catch(err => alert("Network Error: " + err));
    },

    setNestedValue: function (obj, path, value) {
        const keys = path.split('.');
        let current = obj;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!current[keys[i]]) current[keys[i]] = {};
            current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;
    },

    setActive: function (type, filename) {
        const payload = { [type]: filename };
        fetch('/info/load_state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(r => r.json())
            .then(() => {
                this.showToast("Active state updated!", "success");
                setTimeout(() => location.reload(), 800);
            })
            .catch(err => alert("Error: " + err));
    },

    createNew: function (type) {
        const name = prompt(`Enter new ${type} filename (without .json):`);
        if (!name) return;

        fetch(`/info/defaults/${type}`)
            .then(r => r.json())
            .then(defaults => {
                return fetch('/info/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type, filename: name, content: defaults })
                });
            })
            .then(r => r.json())
            .then(() => {
                this.showToast("Created successfully!", "success");
                setTimeout(() => location.reload(), 1000);
            })
            .catch(err => alert("Error: " + err));
    },

    deleteFile: function (type, filename) {
        if (!confirm(`Delete ${filename}? This cannot be undone.`)) return;
        this.showToast("NOT IMPLEMENTED", "info");
        //fetch(`/info/delete_config/${type}/${filename}`, { method: 'DELETE' })
        //    .then(() => location.reload())
        //    .catch(err => alert("Error: " + err));
    },

    showToast: function (message, type) {
        const toast = document.createElement('div');
        // Bootstrap alert classes
        toast.className = `alert alert-${type} shadow-lg border-0 fade show`;
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '10000';
        toast.style.minWidth = '300px';

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

// Initialize
SetupManager.init();
