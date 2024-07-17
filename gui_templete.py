import tkinter as tk
from tkinter import ttk

class GUI:
    
    option_complexity_options = [
        "Vanilla",
        "Exotic"
    ]

    vanilla_options = [
        "European",
        "American"
    ]

    exotic_options = [
        "Binary",
        "Asian",
        "Bermudan",
        "Barrier",
        "Cliquet",
        "Lookback",
        "Rainbow",
        "Basket"
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1200x800")

        # Frame in col 1 that spans all three rows
        self.input_frame = tk.Frame(master=self.root, width=200, height=300)
        self.input_frame.grid(row=0, column=0, rowspan=3, columnspan=2, sticky="nsew")

        # Frame in row 1 that spans columns 3 - 5
        self.output_top_frame = tk.Frame(master=self.root, width=300, height=100)
        self.output_top_frame.grid(row=0, column=2, columnspan=3, sticky="nsew")

        # Frame in row 2-3 that spans columns 3 - 5
        self.output_bottom_frame = tk.Frame(master=self.root, width=300, height=200)
        self.output_bottom_frame.grid(row=1, column=2, rowspan=2, columnspan=3, sticky="nsew")

        # Configure row and column weights
        for i in range(3):
            self.root.grid_rowconfigure(i, weight=1)
        for j in range(5):
            self.root.grid_columnconfigure(j, weight=1)

        ## Input frame

        # Option-type selection
        ttk.Label(self.input_frame, text="Option Specification:").grid(column=0, row=0)
        self.option_type_var = tk.StringVar(self.input_frame)
        option_complexity_dropdown = ttk.Combobox(self.input_frame, values=self.option_complexity_options, 
            textvariable=self.option_type_var, state='readonly')
        option_complexity_dropdown.grid(column=1, row=0)
        option_complexity_dropdown.current(0)

        self.option_type_var.trace_add('write', self.update_specific_options)

        ttk.Label(self.input_frame, text="Option Type:").grid(column=0, row=1)
        self.specific_option_var = tk.StringVar(self.input_frame)
        self.specific_option_dropdown = ttk.Combobox(self.input_frame, textvariable=self.specific_option_var, state='readonly')
        self.specific_option_dropdown.grid(column=1, row=1)

        ## Taking of underlying price, strike, interest rate, volatility, TTE, Dividend yield

        ttk.Label(self.input_frame, text="Underlying Price:").grid(column=0, row=2)
        self.underlyingPrice_var = tk.StringVar(self.input_frame)
        self.entry_underlyingPrice = ttk.Entry(self.input_frame, textvariable=self.underlyingPrice_var, width=10)
        self.entry_underlyingPrice.grid(column=1, row=2, sticky=(tk.W, tk.E))
        self.entry_underlyingPrice.bind("<KeyRelease>", lambda event: self.validate_input(self.underlyingPrice_var))

        ttk.Label(self.input_frame, text="Strike Price:").grid(column=0, row=3)
        self.strikePrice_var = tk.StringVar(self.input_frame)
        self.entry_strikePrice = ttk.Entry(self.input_frame, textvariable=self.strikePrice_var, width=10)
        self.entry_strikePrice.grid(column=1, row=3, sticky=(tk.W, tk.E))
        self.entry_strikePrice.bind("<KeyRelease>", lambda event: self.validate_input(self.strikePrice_var))

        ttk.Label(self.input_frame, text="Interest Rate:").grid(column=0, row=4)
        self.interestRate_var = tk.StringVar(self.input_frame)
        self.entry_interestRate = ttk.Entry(self.input_frame, textvariable=self.interestRate_var, width=10)
        self.entry_interestRate.grid(column=1, row=4, sticky=(tk.W, tk.E))
        self.entry_interestRate.bind("<KeyRelease>", lambda event: self.validate_input(self.interestRate_var))
        self.percent_label = ttk.Label(self.input_frame, text="%")
        self.percent_label.grid(column=2, row=4, sticky=tk.W)

        ttk.Label(self.input_frame, text="Volatility:").grid(column=0, row=5)
        self.volatility_var = tk.StringVar(self.input_frame)
        self.entry_volatility = ttk.Entry(self.input_frame, textvariable=self.volatility_var, width=10)
        self.entry_volatility.grid(column=1, row=5, sticky=(tk.W, tk.E))
        self.entry_volatility.bind("<KeyRelease>", lambda event: self.validate_input(self.volatility_var))
        self.percent_label = ttk.Label(self.input_frame, text="%")
        self.percent_label.grid(column=2, row=5, sticky=tk.W)

        ttk.Label(self.input_frame, text="Dividend Yield:").grid(column=0, row=6)
        self.dividend_var = tk.StringVar(self.input_frame)
        self.entry_dividend = ttk.Entry(self.input_frame, textvariable=self.dividend_var, width=10)
        self.entry_dividend.grid(column=1, row=6, sticky=(tk.W, tk.E))
        self.entry_dividend.bind("<KeyRelease>", lambda event: self.validate_input(self.dividend_var))
        self.percent_label = ttk.Label(self.input_frame, text="%")
        self.percent_label.grid(column=2, row=6, sticky=tk.W)

        ttk.Label(self.input_frame, text="Time to Expiration:").grid(column=0, row=7)
        self.TTE_var = tk.StringVar(self.input_frame)
        self.entry_TTE = ttk.Entry(self.input_frame, textvariable=self.TTE_var, width=10)
        self.entry_TTE.grid(column=1, row=7, sticky=(tk.W, tk.E))
        self.entry_TTE.bind("<KeyRelease>", lambda event: self.validate_input(self.TTE_var))


        # Initialize the specific options dropdown
        self.update_specific_options()
        self.root.mainloop()

    def update_specific_options(self, *args):
        selected_type = self.option_type_var.get()
        if selected_type == "Vanilla":
            self.specific_option_dropdown['values'] = self.vanilla_options
        elif selected_type == "Exotic":
            self.specific_option_dropdown['values'] = self.exotic_options
        else:
            self.specific_option_dropdown['values'] = []

    def validate_input(self, var):
        value = var.get()
        if not self.is_valid_number(value):
            var.set(''.join(filter(self.is_valid_char, value)))

    def is_valid_char(self, char):
        return char.isdigit() or char == '.'

    def is_valid_number(self, value):
        # Allow only digits and a single decimal point
        if value.count('.') > 1:
            return False
        if not value.replace('.', '', 1).isdigit():
            return False
        return True

if __name__ == "__main__":
    app = GUI()
