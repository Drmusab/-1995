---
title: {{ title }}
category: {{ category }}
language: {{ language }}
created: {{ created_at }}
tags: [{% for tag in tags %}{{ tag }}{% if not loop.last %}, {% endif %}{% endfor %}]
---

# {{ title }}

{% if summary %}
- **Summary**
  - {{ summary }}

{% endif %}
{% if key_points %}
- **Key Points**
{% for point in key_points %}  - {{ point }}
{% endfor %}

{% endif %}
{% if action_items %}
- **Action Items**
{% for item in action_items %}  - TODO {{ item }}
{% endfor %}

{% endif %}
{% if definitions %}
- **Definitions**
{% for term, definition in definitions.items %}  - **{{ term }}**: {{ definition }}
{% endfor %}

{% endif %}
- **Transcription**
{% for line in transcription.split('\n') %}{% if line.strip() %}  - {{ line.strip() }}
{% endif %}{% endfor %}