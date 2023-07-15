from unidecode import unidecode

__all__ = ['Englishify']

class Englishify:
    """
    A class for transliteration.
    """
    
    def __init__(self, user_input:str or list = ''):
        """

        Parameters
        ----------
        user_input : str or list, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.

        """
        self.user_input = user_input
        
        
    def __call__(self):
        """
        Class callable method
        
        Returns
        -------
        TYPE
            DESCRIPTION: Convert unicode to the nearest ASCII equivalent.
        """
        return self.transliteration()
    

    def transliteration(self):
        if type(self.user_input) == str:
            return unidecode(self.user_input)
        elif type(self.user_input) == list:
            return [unidecode(s) for s in self.user_input]
            
