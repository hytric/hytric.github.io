---
title: "CV"
permalink: /categories/CV/
layout: category
author_profile: true
taxonomy: Algorithm
---

[Download my CV PDF](https://hytric.github.io/assets/CV/CV.pdf){:target="_blank"}

{% assign posts = site.tags['Algorithm'] %}
{% for post in posts %}
  {% if post.url contains "%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0" %}
    {% include archive-single.html type=page.entries_layout %}
  {% endif %}
{% endfor %}