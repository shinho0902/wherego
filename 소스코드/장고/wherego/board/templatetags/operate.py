from django import template
register = template.Library()

@register.filter
def make_num(counter,page):
    return (counter + (page-1)*5)

@register.filter
def make_page(num,posts):
    return (num//posts+1)
