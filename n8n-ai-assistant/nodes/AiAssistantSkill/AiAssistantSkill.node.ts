import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class AiAssistantSkill implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'AI Assistant Skill',
		name: 'aiAssistantSkill',
		icon: 'file:aiassistant.svg',
		group: ['ai'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Execute AI Assistant skills and capabilities',
		defaults: {
			name: 'AI Assistant Skill',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'aiAssistantApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.baseUrl}}/api/v1',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
			},
		},
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'List Skills',
						value: 'listSkills',
						description: 'Get list of available skills',
						action: 'List available skills',
					},
					{
						name: 'Execute Skill',
						value: 'executeSkill',
						description: 'Execute a specific skill',
						action: 'Execute a skill',
					},
					{
						name: 'Get Skill Info',
						value: 'getSkillInfo',
						description: 'Get information about a specific skill',
						action: 'Get skill information',
					},
					{
						name: 'Note Taking',
						value: 'noteTaking',
						description: 'Use note-taking capabilities',
						action: 'Take notes',
					},
				],
				default: 'listSkills',
			},
			// Execute Skill Parameters
			{
				displayName: 'Skill Name',
				name: 'skillName',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['executeSkill', 'getSkillInfo'],
					},
				},
				default: '',
				placeholder: 'Enter skill name',
				description: 'Name of the skill to execute or get info about',
			},
			{
				displayName: 'Skill Parameters',
				name: 'skillParameters',
				type: 'json',
				displayOptions: {
					show: {
						operation: ['executeSkill'],
					},
				},
				default: '{}',
				description: 'Parameters to pass to the skill as JSON object',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['executeSkill', 'noteTaking'],
					},
				},
				default: '',
				description: 'Session ID for context (optional)',
			},
			// Note Taking Parameters
			{
				displayName: 'Note Operation',
				name: 'noteOperation',
				type: 'options',
				displayOptions: {
					show: {
						operation: ['noteTaking'],
					},
				},
				options: [
					{
						name: 'Create Note',
						value: 'createNote',
					},
					{
						name: 'Update Note',
						value: 'updateNote',
					},
					{
						name: 'Get Note',
						value: 'getNote',
					},
					{
						name: 'List Notes',
						value: 'listNotes',
					},
					{
						name: 'Delete Note',
						value: 'deleteNote',
					},
					{
						name: 'Search Notes',
						value: 'searchNotes',
					},
					{
						name: 'Export Note',
						value: 'exportNote',
					},
				],
				default: 'createNote',
				description: 'Note operation to perform',
			},
			{
				displayName: 'Note Content',
				name: 'noteContent',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['createNote', 'updateNote'],
					},
				},
				default: '',
				placeholder: 'Enter note content',
				description: 'Content of the note',
			},
			{
				displayName: 'Note Title',
				name: 'noteTitle',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['createNote', 'updateNote'],
					},
				},
				default: '',
				placeholder: 'Enter note title',
				description: 'Title of the note',
			},
			{
				displayName: 'Note ID',
				name: 'noteId',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['updateNote', 'getNote', 'deleteNote', 'exportNote'],
					},
				},
				default: '',
				description: 'ID of the note to operate on',
			},
			{
				displayName: 'Category',
				name: 'category',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['createNote', 'updateNote', 'listNotes'],
					},
				},
				default: '',
				description: 'Category for the note',
			},
			{
				displayName: 'Search Query',
				name: 'searchQuery',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['searchNotes'],
					},
				},
				default: '',
				placeholder: 'Enter search query',
				description: 'Query to search for notes',
			},
			{
				displayName: 'Export Format',
				name: 'exportFormat',
				type: 'options',
				displayOptions: {
					show: {
						operation: ['noteTaking'],
						noteOperation: ['exportNote'],
					},
				},
				options: [
					{
						name: 'Markdown',
						value: 'markdown',
					},
					{
						name: 'JSON',
						value: 'json',
					},
					{
						name: 'Plain Text',
						value: 'text',
					},
				],
				default: 'markdown',
				description: 'Format to export the note in',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const operation = this.getNodeParameter('operation', i) as string;

			let responseData;

			try {
				if (operation === 'listSkills') {
					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: '/skills',
							json: true,
						},
					);
				} else if (operation === 'executeSkill') {
					const skillName = this.getNodeParameter('skillName', i) as string;
					const skillParametersString = this.getNodeParameter('skillParameters', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;

					let skillParameters = {};
					if (skillParametersString) {
						try {
							skillParameters = JSON.parse(skillParametersString);
						} catch (error) {
							throw new NodeOperationError(this.getNode(), 'Invalid JSON in skill parameters');
						}
					}

					const body = {
						skill_name: skillName,
						parameters: skillParameters,
						session_id: sessionId || undefined,
					};

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'POST',
							url: '/skills/execute',
							body,
							json: true,
						},
					);
				} else if (operation === 'getSkillInfo') {
					const skillName = this.getNodeParameter('skillName', i) as string;

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: `/skills/${skillName}`,
							json: true,
						},
					);
				} else if (operation === 'noteTaking') {
					const noteOperation = this.getNodeParameter('noteOperation', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;

					if (noteOperation === 'createNote') {
						const content = this.getNodeParameter('noteContent', i) as string;
						const title = this.getNodeParameter('noteTitle', i) as string;
						const category = this.getNodeParameter('category', i) as string;

						const body = {
							content,
							title,
							category: category || undefined,
							session_id: sessionId || undefined,
						};

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'POST',
								url: '/notes',
								body,
								json: true,
							},
						);
					} else if (noteOperation === 'updateNote') {
						const noteId = this.getNodeParameter('noteId', i) as string;
						const content = this.getNodeParameter('noteContent', i) as string;
						const title = this.getNodeParameter('noteTitle', i) as string;
						const category = this.getNodeParameter('category', i) as string;

						const body = {
							content,
							title,
							category: category || undefined,
						};

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'PUT',
								url: `/notes/${noteId}`,
								body,
								json: true,
							},
						);
					} else if (noteOperation === 'getNote') {
						const noteId = this.getNodeParameter('noteId', i) as string;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'GET',
								url: `/notes/${noteId}`,
								json: true,
							},
						);
					} else if (noteOperation === 'listNotes') {
						const category = this.getNodeParameter('category', i) as string;

						const params = new URLSearchParams();
						if (category) {
							params.append('category', category);
						}
						if (sessionId) {
							params.append('session_id', sessionId);
						}

						const url = `/notes${params.toString() ? '?' + params.toString() : ''}`;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'GET',
								url,
								json: true,
							},
						);
					} else if (noteOperation === 'deleteNote') {
						const noteId = this.getNodeParameter('noteId', i) as string;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'DELETE',
								url: `/notes/${noteId}`,
								json: true,
							},
						);
					} else if (noteOperation === 'searchNotes') {
						const searchQuery = this.getNodeParameter('searchQuery', i) as string;

						const params = new URLSearchParams({
							q: searchQuery,
						});

						if (sessionId) {
							params.append('session_id', sessionId);
						}

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'GET',
								url: `/notes/search?${params.toString()}`,
								json: true,
							},
						);
					} else if (noteOperation === 'exportNote') {
						const noteId = this.getNodeParameter('noteId', i) as string;
						const exportFormat = this.getNodeParameter('exportFormat', i) as string;

						const body = {
							format: exportFormat,
						};

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'POST',
								url: `/notes/${noteId}/export`,
								body,
								json: true,
							},
						);
					}
				}

				const executionData = this.helpers.constructExecutionMetaData(
					this.helpers.returnJsonArray(responseData as INodeExecutionData),
					{ itemData: { item: i } },
				);

				returnData.push(...executionData);
			} catch (error) {
				if (this.continueOnFail()) {
					const executionData = this.helpers.constructExecutionMetaData(
						this.helpers.returnJsonArray({ error: error.message }),
						{ itemData: { item: i } },
					);
					returnData.push(...executionData);
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}