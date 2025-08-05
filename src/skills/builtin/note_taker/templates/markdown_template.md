# {{ title }}

## Metadata

| Field | Value |
|-------|-------|
| Created | {{ created_at }} |
| Category | {{ category }} |
| Language | {{ language }} |
| Word Count | {{ word_count }} |
| Duration | {{ duration }}s |
| Confidence | {{ confidence }} |
{% if tags %}| Tags | {% for tag in tags %}`{{ tag }}` {% endfor %}|{% endif %}

{% if audio_path %}
## Audio

📎 **Audio File:** `{{ audio_path }}`
{% endif %}

{% if summary %}
## Summary

{{ summary }}
{% endif %}

{% if key_points %}
## Key Points

{% for point in key_points %}- {{ point }}
{% endfor %}
{% endif %}

{% if action_items %}
## Action Items

{% for item in action_items %}- [ ] {{ item }}
{% endfor %}
{% endif %}

{% if definitions %}
## Definitions

{% for term, definition in definitions.items() %}**{{ term }}:** {{ definition }}

{% endfor %}
{% endif %}

## Full Transcription

```
{{ transcription }}
```