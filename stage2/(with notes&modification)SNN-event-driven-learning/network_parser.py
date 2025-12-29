import yaml


class parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """
    def __init__(self, path):
        # 读取yaml文件
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    # Allow dictionary like access
    def __getitem__(self, key):
        # 可以通过键访问参数
        return self.parameters[key]

    def save(self, filename):
        # 将参数保存到yaml文件，使用dump方法
        # yaml.dump 可以序列化并写入到指定的 filename 文件中，实现配置的保存或备份
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)
