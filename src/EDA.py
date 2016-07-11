
# coding: utf-8

# ## Describing Variables

# In[6]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[7]:

# import all 21 csvs

bill_of_materials_df = pd.read_csv('../competition_data/bill_of_materials.csv')
comp_adaptor_df = pd.read_csv('../competition_data/comp_adaptor.csv')
comp_boss_df = pd.read_csv('../competition_data/comp_boss.csv')
comp_elbow_df = pd.read_csv('../competition_data/comp_elbow.csv')
comp_float_df = pd.read_csv('../competition_data/comp_float.csv')
comp_hfl_df = pd.read_csv('../competition_data/comp_hfl.csv')
comp_nut_df = pd.read_csv('../competition_data/comp_nut.csv')
comp_other_df = pd.read_csv('../competition_data/comp_other.csv')
comp_sleeve_df = pd.read_csv('../competition_data/comp_sleeve.csv')
comp_straight_df = pd.read_csv('../competition_data/comp_straight.csv')
comp_tee_df = pd.read_csv('../competition_data/comp_tee.csv')
comp_threaded_df = pd.read_csv('../competition_data/comp_threaded.csv')
components_df = pd.read_csv('../competition_data/components.csv')
specs_df = pd.read_csv('../competition_data/specs.csv')
test_set_df = pd.read_csv('../competition_data/test_set.csv')
train_set_df = pd.read_csv('../competition_data/train_set.csv')
tube_end_form_df = pd.read_csv('../competition_data/tube_end_form.csv')
tube_df = pd.read_csv('../competition_data/tube.csv')
type_component_df = pd.read_csv('../competition_data/type_component.csv')
type_connection_df = pd.read_csv('../competition_data/type_connection.csv')
type_end_form_df = pd.read_csv('../competition_data/type_end_form.csv')


# In[5]:

train_set_df


# In[104]:

bill_of_materials_df.head()


# In[105]:

# bill_of_materials variables

# tube_assembly_id: ID
# component_id_1: ID
# quantity_1: number of components needed for assembly (conceptually integers, practically floats)
# component_id_2: ID
# quantity_2: number of components needed for assembly (conceptually integers, practically floats)
# component_id_3: ID
# quantity_3: number of components needed for assembly (conceptually integers, practically floats)
# component_id_4: ID
# quantity_4: number of components needed for assembly (conceptually integers, practically floats)
# component_id_4: ID
# quantity_4: number of components needed for assembly (conceptually integers, practically floats)
# component_id_5: ID
# quantity_5: number of components needed for assembly (conceptually integers, practically floats)
# component_id_6: ID
# quantity_6: number of components needed for assembly (conceptually integers, practically floats)
# component_id_7: ID
# quantity_7: number of components needed for assembly (conceptually integers, practically floats)
# component_id_8: ID
# quantity_8: number of components needed for assembly (conceptually integers, practically floats)


# In[106]:

bill_of_materials_df.info()


# In[107]:

comp_adaptor_df.head()


# In[108]:

# comp_[type] variables: 

# 1. we're not sure what each column physically refers to or the length units.
# 2. 9999 could be another NaN

# There appear to be 11 basic component types, with further broken down subtypes. Each type has type-specific
# variables that may not apply to other component types.


# In[109]:

specs_df.head()


# In[110]:

# specs_df

# list of specifications for components of each tube assembly (may be 0-10)


# In[111]:

test_set_df.head(n = 5)


# In[112]:

# test_set_df

#id: (integer) observation index
#tube_assembly_id: (string) ID
#supplier: (string) ID
#quote_date: (string) when price was quoted by supplier 
#annual_usage: (int) estimate of how many tube assemblies will be purchased in a given year
#min_order_quantity: (integer) number of assemblies required, at minimum, for non-bracket pricing
#bracket_pricing: (string) does this assembly have bracket-pricing or not?
#quantity: (integer) how many assemblies are sought for purchase?


# In[113]:

train_set_df.head(n = 5)


# In[114]:

print train_set_df['quote_date'].sort_values()[8449]
print test_set_df['quote_date'].sort_values()


# In[115]:

# train_set_df

#id: (integer) observation index
#tube_assembly_id: (string) ID
#supplier: (string) ID
#quote_date: (string) when price was quoted by supplier 
#annual_usage: (int) estimate of how many tube assemblies will be purchased in a given year
#min_order_quantity: (integer) number of assemblies required, at minimum, for non-bracket pricing
#bracket_pricing: (string) does this assembly have bracket-pricing or not?
#quantity: (integer) how many assemblies are sought for purchase?
#cost: (float) price per assembly in dollars.

train_set_df['tube_assembly_id'] = pd.Series(train_set_df['tube_assembly_id'], dtype = 'category')
train_set_df['supplier'] = pd.Series(train_set_df['supplier'], dtype = 'category')
train_set_df['bracket_pricing'] = pd.Series(train_set_df['bracket_pricing'], dtype = 'category')

df = pd.to_datetime(train_set_df['quote_date'])
df = df.to_frame()
df['first_date'] = pd.Timestamp('19820922')
train_set_df['quote_date'] = (df['quote_date'] - df['first_date']).dt.days


# In[116]:

train_set_df.head()


# In[117]:

tube_end_form_df


# In[118]:

# tube_end_form_df
# 9999 might be 'other' category

# end_form_id: (string) ID
# forming: (string) forming or not(?)


# In[119]:

tube_df['tube_assembly_id'] = pd.Series(tube_df['tube_assembly_id'], dtype = 'category')
tube_df['material_id'] = pd.Series(tube_df['material_id'], dtype = 'category')
tube_df['end_a_1x'] = pd.Series(tube_df['end_a_1x'], dtype = 'category')
tube_df['end_a_2x'] = pd.Series(tube_df['end_a_2x'], dtype = 'category')
tube_df['end_x_1x'] = pd.Series(tube_df['end_x_1x'], dtype = 'category')
tube_df['end_x_2x'] = pd.Series(tube_df['end_x_2x'], dtype = 'category')
tube_df['end_a'] = pd.Series(tube_df['end_a'], dtype = 'category')
tube_df['end_x'] = pd.Series(tube_df['end_x'], dtype = 'category')
# replace 9999.0 entries in bend_radius column with np.nan entries
tube_df = tube_df.replace(9999.0, np.nan)
tube_df = tube_df.replace('9999', 'other')
np.shape(tube_df)

## deleting rows with tube length of zero
#tube_df[tube_df['length'] == 0.0].index
#tube_df = tube_df.drop(tube_df[tube_df['length'] == 0.0].index)


# In[120]:

tube_df.describe()


# In[121]:

# tube_df

#tube_assembly_id: (string) ID
#material_id: (string) material specification for the tube
#diameter: (float) tube diameter
#wall: (float) wall thickness
#length: (float) tube length
#num_bends: (integer) number of bends in tube
#bend_radius: (float) (guess) radius of all bends in tube
#end_a_1x: (string) length of end a is less than 1x diameter
#end_a_2x: (string) length of end a is less than 2x diameter
#end_x_1x: (string) length of end x is less than 1x diameter
#end_x_2x: (string) length of end x is less than 2x diameter
#end_a: (string) end form of end a
#end_x: (string) end form of end x
#num_boss: (integer) number of bosses
#num_bracket: (integer) number of brackets
#other: (integer) other tube prep required steps


# In[122]:

type_component_df


# In[123]:

# type_component_df

# component_type_id: (string) ID
# name: (string) defines how component is attached to tube


# In[124]:

type_connection_df


# In[125]:

# type_connection_df
# refer only to adaptors

# connection_type_id: (string) ID
# name: (string) connection type description


# In[126]:

type_end_form_df


# In[127]:

# type_end_form_df

#end_form_id = (string) ID
#name = (string) end form description


# In[128]:

bill_comp_types_df = bill_of_materials_df.iloc[:, (1,3,5,7,9,11,13,15)]
bill_comp_types_logical_df = ~bill_comp_types_df.isnull()
component_series = bill_comp_types_logical_df.sum(axis = 1)
bins = range(9)
plt.hist(component_series, bins = bins, align = 'left')
# almost half of all tube assemblies have exactly 2 types of components


# In[ ]:




# In[129]:

(sum(component_series == 0) + sum(component_series == 1) + sum(component_series == 2)  + sum(component_series == 3))/float(component_series.count())
# Over 97% of all tube assemblies have components from 0-3


# In[130]:

bill_comp_quants_df = bill_of_materials_df.iloc[:, (2,4,6,8,10,12,14,16)]
quants_series = bill_comp_quants_df.sum(axis = 1)
bins = range(14)
plt.hist(quants_series, bins = bins, align = 'left')


# In[131]:

(sum(quants_series == 0) + sum(quants_series == 1) + sum(quants_series == 2)  + sum(quants_series == 3) + sum(quants_series == 4)) / float(np.shape(bill_of_materials_df)[0])
# Over 98% of all tube assemblies have 0-4 total components


# In[132]:

bill_of_materials_df.describe()


# In[133]:

tube_df.head()


# In[134]:

tube_df.describe()
#


# In[135]:

diameter_series = tube_df['diameter']
bins = range(30)
plt.hist(diameter_series, bins = 30, align = 'left')
# smaller diameters, less than 30, are dominant


# In[136]:

wall_series = tube_df['wall']
bins = range(30)
plt.hist(wall_series, bins = 15, align = 'left')
# smaller wall thicknesses, less than 1.5, are dominant.


# In[137]:

length_series = tube_df['length']
bins = range(30)
plt.hist(length_series, bins = 15, align = 'mid')
# shorter tubes, less than 250, are dominant.


# In[138]:

tube_df.describe(include = 'all')
# num_boss, num_bracket, and other are mostly 0, with small maxima.


# In[139]:

sns.pairplot(tube_df[['diameter', 'wall', 'length']])


# In[140]:

tube_df.corr()


# In[141]:

specs_df.head()


# In[142]:

specs_only_df = specs_df.iloc[:, 1:11]
specs_logical_df = ~specs_only_df.isnull()
spec_totals = specs_logical_df.sum(axis = 1)
bins = range(11)
plt.hist(spec_totals, bins = bins, align = 'left')
# almost half of all tube assemblies have exactly 2 types of components


# In[143]:

train_set_df.describe(include = 'all')


# In[144]:

sns.pairplot(train_set_df)


# In[145]:

tube_df.head()


# In[146]:

#left join with left = train_set_df and right = tube_df

result_1 = pd.merge(train_set_df,
                    tube_df,
                    left_on = 'tube_assembly_id',
                    right_on = 'tube_assembly_id',
                    how='left',
                    sort=False)


# In[147]:

# left join with left = result_1 (train and tube) and right = specs_with_totals

specs_with_totals_df = specs_df.copy()
specs_with_totals_df['spec_totals'] = spec_totals

result_2 = result_1.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])


# In[148]:

# left join with left = result_2 (train, tube, and specs_with_totals) and right = bill_of_materials_summary_df

bill_of_materials_summary_df = bill_of_materials_df.copy()
bill_of_materials_summary_df['type_totals'] = component_series
bill_of_materials_summary_df['component_totals'] = quants_series

result_3 = result_2.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])


# In[149]:

pd.options.display.max_columns = 50


# In[150]:

basic_df = result_3.copy()


# In[151]:

(~basic_df.isnull()).sum()


# In[152]:

type(basic_df['material_id'][23228])


# In[153]:

np.isnan(6.0)


# In[154]:

a = basic_df['tube_assembly_id'].unique()


# In[155]:

b = test_set_df['tube_assembly_id'].unique()


# In[156]:

# training and test sets are disjoint for tube_assembly_id
np.intersect1d(a, b)


# In[157]:

basic_df


# In[158]:

basic_df.describe(include = 'all')


# In[159]:

# 45 unique suppliers are in both train and test dfs
a = train_set_df['supplier']
b = test_set_df['supplier']
len(np.intersect1d(a, b))


# In[160]:

# get rid of tube_assembly_id column in basic_df
basic_df.drop(labels = 'tube_assembly_id', axis = 1, inplace = True)


# In[161]:

# replace supplier column with 57 dummy variable columns
supplier_series = basic_df['supplier']
supplier_dummy_df = pd.get_dummies(data = supplier_series, prefix = 'supp', prefix_sep = '_')
np.shape(supplier_dummy_df)
basic_df.drop(labels = 'supplier', axis = 1, inplace = True)


# In[162]:

basic_df = pd.concat(objs = [basic_df, supplier_dummy_df], axis = 1)


# In[163]:

# convert N/Y bracket pricing column to 0/1 
bracket_pricing_series = basic_df['bracket_pricing']
basic_df['bracket_pricing'] = bracket_pricing_series.replace(['No', 'Yes'], [0, 1])


# In[164]:

basic_df


# In[165]:

# replace material_id column with 18 dummy variable columns, including NA
material_id_series = basic_df['material_id']
material_id_dummy_df = pd.get_dummies(data = material_id_series, prefix = 'mat', prefix_sep = '_', dummy_na = True)
np.shape(material_id_dummy_df)
basic_df.drop(labels = 'material_id', axis = 1, inplace = True)


# In[166]:

basic_df = pd.concat(objs = [basic_df, material_id_dummy_df], axis = 1)


# In[167]:

# convert end_a_1x, end_a_2x, end_x_1x, end_x_2x columns to 0/1. 
basic_df['end_a_1x'] = basic_df['end_a_1x'].replace(['N', 'Y'], [0, 1])
basic_df['end_a_2x'] = basic_df['end_a_2x'].replace(['N', 'Y'], [0, 1])
basic_df['end_x_1x'] = basic_df['end_x_1x'].replace(['N', 'Y'], [0, 1])
basic_df['end_x_2x'] = basic_df['end_x_2x'].replace(['N', 'Y'], [0, 1])


# In[168]:

# replace end_a column with 25 indicator variable columns
end_a_series = basic_df['end_a']
end_a_dummy_df = pd.get_dummies(data = end_a_series, prefix = 'end_a', prefix_sep = '_')
np.shape(end_a_dummy_df)
basic_df.drop(labels = 'end_a', axis = 1, inplace = True)


# In[169]:

basic_df = pd.concat(objs = [basic_df, end_a_dummy_df], axis = 1)


# In[170]:

# replace end_x column with 24 indicator variable columns
end_x_series = basic_df['end_x']
end_x_dummy_df = pd.get_dummies(data = end_x_series, prefix = 'end_x', prefix_sep = '_')
np.shape(end_x_dummy_df)
basic_df.drop(labels = 'end_x', axis = 1, inplace = True)


# In[171]:

basic_df = pd.concat(objs = [basic_df, end_x_dummy_df], axis = 1)


# In[172]:

pd.options.display.max_columns = 150
np.shape(basic_df)


# In[173]:

np.where(np.isnan(basic_df))


# In[174]:

# now remove all rows with nan values for bend_radius in basic_df
basic_df.dropna(axis = 0, how = 'any', inplace = True)


# In[175]:

np.shape(basic_df)


# In[176]:

np.where(np.isnan(basic_df))


# In[177]:

basic_df.to_csv(path_or_buf = 'basic_df.csv')


# In[178]:

plt.hist(basic_df['cost'], bins = range(0, 1000, 10))


# In[179]:

sum(basic_df['bracket_pricing'])/float(np.shape(basic_df)[0])


# In[180]:

basic_df.head()


# In[181]:

np.shape(basic_df)


# In[ ]:



