from django import forms


class Subscribe(forms.Form):
    Email = forms.CharField()

    def __str__(self):
        return self.Email
