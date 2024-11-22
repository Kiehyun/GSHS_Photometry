#%%
class addition:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        result = self.a + self.b
        return result

 #%%
class calculator(addition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cal = 'I am a Calculator'

 #%%
tool1=calculator(1,2)
tool1.add()

#%%
tool2=calculator(a=1, b=2)
tool2.add()

# %%
class addition:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        result = self.a + self.b
        return result
 
 # %%
class calculator(addition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cal = 'I am a Calculator'
 
# %%
tool1=calculator()
tool1.add()

# %%
tool1=calculator(5, 6)
tool1.add()

# %%
print("tool1.a:", tool1.a)
print("tool1.b:", tool1.b)




# %%
