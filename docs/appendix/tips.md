# Tips
Are you here for some tips on writing? Here are some resources and formatting tricks you can play around with markdown writing!

## Markdown Tricks
### Make use of the built-in navigation
Look at the right side of the page. When you have multiple sections in a page, you can see the section break downs on the right. It comes for free!

### Reference

Anything within the `docs/` folder are cross-reference-able.

For example, [click me](../arch/index.md) (`[click me](../arch/index.md)`)will take you to the Architecture page.


### Attachment

Similarly, this is also how you include a picture.

!!! example Include a Cat Picture
    ```
    ![cat](tips_md/example.jpg)
    ```

![cat](tips_md/example.jpg)


## Admonition
!!! tip ""
    Admonition is powerful to make your documentation vivid.

??? success
    And it's cool!

Check [here](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) for a full list of supported admonition.
