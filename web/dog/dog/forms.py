from django import forms

class ImgForm(forms.Form):
    addition = (
    ('dog ear', 'dog ear'),
    ('elf crown', 'elf crown'),
)
    image = forms.ImageField()
    overlay = forms.ChoiceField(choices = addition)
    def clean_user_file(self, *args, **kwargs):
        cleaned_data = super(ImgForm,self).clean()
        image = cleaned_data.get("Image")
 
        if image:
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError("File is too big.")
 
        return image
class UrlForm(forms.Form):
    addition = (
    ('dog ear', 'dog ear'),
    ('elf crown', 'elf crown'),
)
    url=forms.URLField()
    overlay = forms.ChoiceField(choices = addition)