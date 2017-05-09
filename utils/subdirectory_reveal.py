import os
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



print(get_immediate_subdirectories(os.getcwd()))


print(get_immediate_subdirectories("C:\opt\jenkins\workspace\d_s2\Chutzpah.4.3.0"))
