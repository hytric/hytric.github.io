---
title: "PaperReview"
permalink: /categories/PaperReview/
layout: category
author_profile: true
taxonomy: PaperReview
---

{% assign posts = site.tags['PaperReview'] %}
{% for post in posts %} 
    {% if post.url contains "%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0" %}
        {% include archive-single.html type=page.entries_layout %}
    {% endif %}
{% endfor %}