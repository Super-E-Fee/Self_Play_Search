class PrefixTool:
    def __init__(self, prefix_path):
        self.prefix_path_head = prefix_path
        
    # def apply_prefix(self, decorated_text):
    #     return self.prefix + ' ' + decorated_text
    def get_prefix(self, prefix_type):

        self.prefix_type = prefix_type
        self.prefix_path = self.prefix_path_head+'/'+self.prefix_type+'.txt'

        try:
            with open(self.prefix_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 去除各种换行符：\n, \r, \r\n
                content = content.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                self.prefix = content
        except FileNotFoundError:
            raise FileNotFoundError(f"File Not Found: {self.prefix_path}")

        return self.prefix