# This document tries to cover many common markdown contents

This is not based on an existing model card and serves to increase test coverage. It also documents differences that may be found after parsing. There is no metainfo section.

## H2

### H3

#### H4

##### H5

###### H6

Parser 'preserves' some "quotation" marks.

Parser doesn’t ‘preserve’ other “quotation” marks.

## Italics

One _way_ of doing it.
Another *way* of doing it.

## Bold

One __way__ of doing it.
Another **way** of doing it.

## Strikethrough

This is ~~not~~ the way.

## Superscript and subscripts

Really just html tags.

E = mc<sup>2</sup>

log<sub>2</sub>

## Bullet lists

Pandoc does not differentiate between different notations, so we always use -, not * or +.

* using
* asterisk

or

- using
- minus
  with line break

or

+ using plus

Finally:

- nesting
  - is
- indeed
  - very
    - possible
  - to achieve

## Ordered lists

1. a normal
2. ordered list

or

1. an ordered
2. list
   1. with
   2. indentation
3. is possible

## Mixed lists

1. it’s
2. possible
   - to
   - mix
3. ordered _and_ unorderd

## TODOs

- [x] This
- [ ] is
- [x] **done**

## Links

[a link](https://skops.readthedocs.io/)

The "title" is not parsed by pandoc

[a link](https://skops.readthedocs.io/ "this disappears")

[a link to a file](./toy-example.md)

References are resolved, so `[1]` below is replaced by the actual link:

[a link with reference][1]

A plain link to https://skops.readthedocs.io/ used inside of text.

[1]: https://skops.readthedocs.io/

## Images

![skops logo](https://github.com/skops-dev/skops/blob/main/docs/images/logo.png)

### Using html

<img src="https://github.com/skops-dev/skops/blob/main/docs/images/logo.png" alt="logo" width="100"/>

## Quotes

> Someone said something importent

> I quote wise words:
> > Someone said something importent

## Tables

| Header 0     | Header 1       |
|--------------|----------------|
| Some content | More content   |
| _Even more_  | This is **it** |

Empty tables are legal

| What now?   |
|-------------|

## Inline code

Some `inline` code.

`A whole line`

## Code blocks

```
A raw

code block
```

With language

```python
def foo():
  return 0
  
def bar():
  return 1
```

## Raw HTML
<p hidden>Cryptids of Revachol:</p>

<dl>
    <dt>Beast of Bodmin</dt>
    <dd>A large feline inhabiting Bodmin Moor.</dd>

    <dt>Morgawr</dt>
    <dd>A sea serpent.</dd>

    <dt>Owlman</dt>
    <dd>A giant owl-like creature.</dd>
</dl>

## Div

The "id" tag may change in order
<div class="warning" somekey key="with value" id="123">
  <p>Divs are possible</p>
</div>

## Line breaks

A text with  
a LineBreak item.
