import React, { useState } from 'react';
import { Save, Plus, Trash2, ChevronRight, ChevronDown, X } from 'lucide-react';

// --- HELPERS (Keep these the same) ---
const getFieldType = (prop: any) => {
    if (!prop) return 'text';
    if (prop.$ref || (prop.allOf && prop.allOf[0].$ref)) return 'nested_model';
    if (prop.type === 'integer' || prop.type === 'number') return 'number';
    if (prop.type === 'boolean') return 'boolean';
    if (prop.type === 'array') return 'array';
    if (prop.type === 'object' && !prop.properties && !prop.$ref) return 'dict';
    return 'text';
};

const resolveRef = (ref: string) => {
    const parts = ref.split('/');
    return parts[parts.length - 1];
};

const CheckIcon = () => (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 3L4.5 8.5L2 6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

// --- RECURSIVE FIELD COMPONENT ---
const FormField = ({ name, schema, value, onChange, definitions, depth = 0 }: any) => {
    const title = schema.title || name;
    const type = getFieldType(schema);
    const description = schema.description;

    // Auto-expand top level only
    const [isExpanded, setIsExpanded] = useState(depth < 1);

    // --- NESTED MODELS ---
    if (type === 'nested_model') {
        const refString = schema.$ref || (schema.allOf && schema.allOf[0].$ref);
        if (!refString) return <div className="text-red-500 text-xs p-2">Invalid Reference</div>;

        const refName = resolveRef(refString);
        const refSchema = definitions[refName];

        if (!refSchema) return <div className="text-red-500 text-xs p-2">Definition {refName} not found</div>;

        const currentVal = value || {};

        return (
            <div className={`mb-4 rounded-lg overflow-hidden border ${depth % 2 === 0 ? 'border-slate-800 bg-slate-950/30' : 'border-slate-700 bg-slate-900/30'}`}>
                <div
                    className={`px-3 py-2 flex items-center justify-between cursor-pointer transition-colors ${depth % 2 === 0 ? 'bg-slate-900 hover:bg-slate-800' : 'bg-slate-800 hover:bg-slate-700'}`}
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    <div className="flex items-center gap-2">
                        {isExpanded ? <ChevronDown size={14} className="text-blue-400" /> : <ChevronRight size={14} className="text-slate-500" />}
                        <span className="text-sm font-semibold text-slate-200">{title}</span>
                    </div>
                </div>

                {isExpanded && (
                    <div className="p-3 space-y-3 border-t border-slate-800/50">
                        {Object.keys(refSchema.properties || {}).map((propKey) => (
                            <FormField
                                key={propKey}
                                name={propKey}
                                schema={refSchema.properties[propKey]}
                                definitions={definitions}
                                value={currentVal[propKey] ?? refSchema.properties[propKey].default}
                                onChange={(val: any) => onChange({ ...currentVal, [propKey]: val })}
                                depth={depth + 1}
                            />
                        ))}
                    </div>
                )}
            </div>
        );
    }

    // --- DICT ---
    if (type === 'dict') {
        const currentDict = value || {};
        const entries = Object.entries(currentDict);

        const updateDictKey = (oldKey: string, newKey: string, val: any) => {
            const newDict: any = {};
            entries.forEach(([k, v]) => {
                if (k === oldKey) newDict[newKey] = val; // Rename key
                else newDict[k] = v;
            });
            onChange(newDict);
        };

        const updateDictVal = (key: string, val: any) => {
            onChange({ ...currentDict, [key]: val });
        };

        const removeKey = (key: string) => {
            const newDict = { ...currentDict };
            delete newDict[key];
            onChange(newDict);
        };

        return (
            <div className="mb-4 p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-medium text-slate-300">{title}</label>
                    <button onClick={() => onChange({ ...currentDict, [`new_key_${entries.length}`]: "" })} className="text-xs flex items-center gap-1 text-blue-400 hover:text-blue-300">
                        <Plus size={12} /> Add
                    </button>
                </div>
                <div className="space-y-2">
                    {entries.map(([k, v]: [string, any], idx) => (
                        <div key={idx} className="flex gap-2 items-center">
                            <input
                                className="w-1/3 bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-300 focus:border-blue-500 outline-none"
                                value={k}
                                onChange={(e) => updateDictKey(k, e.target.value, v)}
                            />
                            <span className="text-slate-600">:</span>
                            <input
                                className="flex-1 bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-blue-500 outline-none"
                                value={v}
                                onChange={(e) => {
                                    const val = !isNaN(Number(e.target.value)) && e.target.value !== '' ? Number(e.target.value) : e.target.value;
                                    updateDictVal(k, val);
                                }}
                            />
                            <button onClick={() => removeKey(k)} className="text-slate-500 hover:text-red-400">
                                <X size={14} />
                            </button>
                        </div>
                    ))}
                    {entries.length === 0 && <div className="text-xs text-slate-600 italic">Empty dictionary</div>}
                </div>
            </div>
        );
    }

    // --- ARRAYS ---
    if (type === 'array') {
        const items = value || [];
        return (
            <div className="mb-4">
                <div className="flex justify-between items-center mb-1.5">
                    <label className="text-sm font-medium text-slate-300">{title}</label>
                    <button
                        onClick={() => onChange([...items, 0])}
                        className="text-xs flex items-center gap-1 text-blue-400 hover:text-blue-300"
                    >
                        <Plus size={12} /> Add
                    </button>
                </div>
                <div className="space-y-1.5 pl-2 border-l-2 border-slate-800">
                    {items.map((item: any, idx: number) => (
                        <div key={idx} className="flex gap-2">
                            <input
                                type="number"
                                className="flex-1 bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-200 focus:border-blue-500 outline-none"
                                value={item}
                                onChange={(e) => {
                                    const newItems = [...items];
                                    newItems[idx] = Number(e.target.value);
                                    onChange(newItems);
                                }}
                            />
                            <button
                                onClick={() => onChange(items.filter((_: any, i: number) => i !== idx))}
                                className="p-1.5 text-slate-500 hover:text-red-400 transition-colors"
                            >
                                <Trash2 size={14} />
                            </button>
                        </div>
                    ))}
                    {items.length === 0 && <div className="text-xs text-slate-600 italic pl-1">No items</div>}
                </div>
            </div>
        );
    }

    // --- PRIMITIVE INPUTS ---
    return (
        <div className="mb-4">
            <div className="flex justify-between items-baseline mb-1.5">
                <label className="block text-sm font-medium text-slate-300">{title}</label>
            </div>

            {type === 'boolean' ? (
                <div
                    onClick={() => onChange(!value)}
                    className="flex items-center gap-3 p-2.5 bg-slate-950 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors group"
                >
                    <div className={`w-4 h-4 rounded flex items-center justify-center border transition-all ${value ? 'bg-blue-600 border-blue-600' : 'bg-slate-900 border-slate-600 group-hover:border-slate-500'}`}>
                        {value && <CheckIcon />}
                    </div>
                    <span className="text-sm text-slate-400 select-none">{value ? 'Enabled' : 'Disabled'}</span>
                </div>
            ) : (
                <div className="relative group">
                    <input
                        type={type === 'number' ? 'number' : 'text'}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
                        value={value ?? ''}
                        placeholder={String(schema.default || '')}
                        onChange={(e) => {
                            const val = type === 'number' ? Number(e.target.value) : e.target.value;
                            onChange(val);
                        }}
                    />
                    {description && <div className="mt-1 text-[10px] text-slate-500 leading-tight break-words">{description}</div>}
                </div>
            )}
        </div>
    );
};

// --- MAIN CONFIG FORM ---
export const ConfigForm = ({ schema, initialData, onSave }: any) => {
    const [formData, setFormData] = useState(initialData || {});
    const definitions = schema.$defs || schema.definitions || {};

    return (
        /* 
           This container takes up 100% of the Modal body.
           We use flex-col to separate the SCROLLABLE form from the FIXED footer.
        */
        <div className="flex flex-col h-full bg-slate-900 text-slate-200">

            {/* 1. SCROLLABLE AREA */}
            <div className="flex-1 overflow-y-auto min-h-0 pr-1 custom-scrollbar">
                <div className="pb-4"> {/* Padding bottom ensures last field isn't flush with footer */}
                    {schema.properties ? (
                        Object.keys(schema.properties).map((key) => (
                            <FormField
                                key={key}
                                name={key}
                                schema={schema.properties[key]}
                                definitions={definitions}
                                value={formData[key] ?? schema.properties[key].default}
                                onChange={(val: any) => setFormData((prev: any) => ({ ...prev, [key]: val }))}
                                depth={0}
                            />
                        ))
                    ) : (
                        <div className="text-slate-500 italic p-4 text-center">No properties found.</div>
                    )}
                </div>
            </div>

            {/* 2. FIXED FOOTER (Outside Scroll View) */}
            <div className="mt-auto pt-4 border-t border-slate-800 bg-slate-900 flex justify-end shrink-0 z-10">
                <button
                    onClick={() => onSave(formData)}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg text-sm font-semibold shadow-lg shadow-blue-900/20 flex items-center gap-2 transition-all hover:translate-y-[-1px] active:scale-95"
                >
                    <Save size={16} /> Save Changes
                </button>
            </div>
        </div>
    );
};
